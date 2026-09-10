#include "kernel_operator.h"
#include "sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_sparse_lightning_indexer_grad_kl_loss.h"
using namespace AscendC;

__aicore__ inline float SliExp(float x) {
    if (x < -87.0f) return 0.0f;
    float z = x * 1.4426950408889634f;
    int n = static_cast<int>(z);
    if (z < static_cast<float>(n)) --n;
    float r = (x - n * 0.693145751953125f) - n * 1.428606765330187e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.1666666666666667f +
        r * (0.0416666666666667f + r * (0.0083333333333333f + r *
        (0.0013888888888889f + r * (0.0001984126984127f + r * 0.0000248015873016f)))))));
    union { uint32_t u; float f; } power;
    power.u = static_cast<uint32_t>(n + 127) << 23;
    return p * power.f;
}
__aicore__ inline float SliLog(float x) {
    union { uint32_t u; float f; } bits;
    bits.f = x;
    int exponent = static_cast<int>(bits.u >> 23) - 127;
    bits.u = (bits.u & 0x7fffffU) | (127U << 23);
    float y = (bits.f - 1.0f) / (bits.f + 1.0f), y2 = y*y;
    float p = 1.0f/17.0f;
    p = 1.0f/15.0f + y2*p; p = 1.0f/13.0f + y2*p;
    p = 1.0f/11.0f + y2*p; p = 1.0f/9.0f + y2*p;
    p = 1.0f/7.0f + y2*p; p = 1.0f/5.0f + y2*p;
    p = 1.0f/3.0f + y2*p; p = 1.0f + y2*p;
    return exponent * 0.6931471805599453f + 2*y*p;
}
__aicore__ inline void SliSum(float value, float &sum, float &correction) {
    float y = value - correction;
    float t = sum + y;
    correction = (t - sum) - y;
    sum = t;
}

// A two-float expansion retains probability residuals through cancellation.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
struct SliPair {
    float hi, lo;
    __aicore__ inline SliPair(float h=0, float l=0) : hi(h), lo(l) {}
    __aicore__ inline float Value() const { return hi+lo; }
};
__aicore__ inline SliPair SliNormalize(float hi, float lo) {
    float sum=hi+lo;
    return SliPair(sum,lo-(sum-hi));
}
__aicore__ inline SliPair operator+(SliPair a, SliPair b) {
    float s=a.hi+b.hi, v=s-a.hi;
    float e=(a.hi-(s-v))+(b.hi-v);
    return SliNormalize(s, e+a.lo+b.lo);
}
__aicore__ inline SliPair operator-(SliPair a, SliPair b) {
    return a+SliPair(-b.hi,-b.lo);
}
__aicore__ inline SliPair operator*(SliPair a, SliPair b) {
    union {float f; uint32_t u;} aa,bb;
    aa.f=a.hi; bb.f=b.hi;
    aa.u &= 0xfffff000U; bb.u &= 0xfffff000U;
    float al=a.hi-aa.f, bl=b.hi-bb.f, p=a.hi*b.hi;
    float e=((aa.f*bb.f-p)+aa.f*bl+al*bb.f)+al*bl;
    return SliNormalize(p,e+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo);
}
__aicore__ inline SliPair operator/(SliPair a, SliPair b) {
    float q=a.hi/b.hi;
    SliPair r=a-b*SliPair(q);
    float q2=(r.hi+r.lo)/b.hi;
    return SliNormalize(q,q2);
}
__aicore__ inline bool operator>(SliPair a, SliPair b) {
    return a.hi>b.hi || (a.hi==b.hi && a.lo>b.lo);
}
__aicore__ inline SliPair SliPairExp(SliPair x) {
    if(x.hi < -87.0f) return SliPair(0);
    float z=x.hi*1.4426950408889634f;
    int n=static_cast<int>(z);
    if(z<static_cast<float>(n)) --n;
    SliPair r=x-SliPair(0.693147182464599609375f,-1.904654323148236e-9f)*SliPair(static_cast<float>(n));
    SliPair term(1),sum(1);
    for(int i=1;i<=16;++i) {
        term=term*r/SliPair(static_cast<float>(i));
        sum=sum+term;
    }
    union {uint32_t u;float f;} power;
    power.u=static_cast<uint32_t>(n+127)<<23;
    return sum*SliPair(power.f);
}
__aicore__ inline SliPair SliPairLog(SliPair x) {
    float seed=SliLog(x.hi);
    SliPair e=SliPairExp(SliPair(seed));
    SliPair residual=(x-e)/e;
    return SliPair(seed)+residual-residual*residual*SliPair(0.5f);
}

template<class T, class W, class QR, class KR>
class SparseGrad {
    GlobalTensor<T> q, k, qi, ki, dq, dk;
    GlobalTensor<W> w, dw;
    GlobalTensor<QR> qr;
    GlobalTensor<KR> kr;
    GlobalTensor<int32_t> indices, ql32, kl32;
    GlobalTensor<int64_t> ql64, kl64;
    GlobalTensor<float> maxima, sums, scratch, losses, loss;
    SparseLightningIndexerGradKlLossTilingData t;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    LocalTensor<float> a, b, product, temporary, result, qiCache, dqCache, keyCache, dkCache, gradTemp, weightOut;
    LocalTensor<uint8_t> raw;
    uint32_t stride;
    uint64_t simOffset, massOffset, logitOffset;

    template<class U>
    __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<U> src, uint64_t offset, uint32_t count) {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        if constexpr (AscendC::IsSameType<U,float>::value) {
            DataCopyPad(dst, src[offset], p, DataCopyPadExtParams<float>{false,0,0,0});
        } else {
            auto input = raw.ReinterpretCast<U>();
            DataCopyPad(input, src[offset], p, DataCopyPadExtParams<U>{false,0,0,0});
            PipeBarrier<PIPE_ALL>();
            Cast(dst, input, RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<class U>
    __aicore__ inline void Store(GlobalTensor<U> dst, uint64_t offset, LocalTensor<float> src, uint32_t count) {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        if constexpr (AscendC::IsSameType<U,float>::value) {
            DataCopyPad(dst[offset], src, p);
        } else {
            auto output = raw.ReinterpretCast<U>();
            Cast(output, src, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(dst[offset], output, p);
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<class A, class B>
    __aicore__ inline float Dot(GlobalTensor<A> x, uint64_t xo, GlobalTensor<B> y, uint64_t yo, uint32_t count) {
        Load(a, x, xo, count); Load(b, y, yo, count);
        Mul(product, a, b, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(result, product, temporary, count);
        PipeBarrier<PIPE_ALL>();
        return result.GetValue(0);
    }
    __aicore__ inline float MainScore(uint32_t batch, uint32_t row, uint32_t head, uint32_t key) {
        uint64_t qo = (static_cast<uint64_t>(batch)*t.queryRows + row)*t.queryHeads + head;
        uint64_t ko = static_cast<uint64_t>(batch)*t.keyRows + key;
        return (Dot(q, qo*t.queryDim, k, ko*t.queryDim, t.queryDim) +
                Dot(qr, qo*t.ropeDim, kr, ko*t.ropeDim, t.ropeDim)) * t.scale;
    }
    __aicore__ inline uint32_t Length(bool query, uint32_t batch) {
        uint32_t type = query ? t.queryLengthType : t.keyLengthType;
        uint32_t bound = query ? t.queryRows : t.keyRows;
        if (!type) return bound;
        int64_t value = type == 1 ? (query ? ql32.GetValue(batch) : kl32.GetValue(batch)) :
                                   (query ? ql64.GetValue(batch) : kl64.GetValue(batch));
        return value < 0 ? 0 : (value > bound ? bound : static_cast<uint32_t>(value));
    }
    __aicore__ inline bool Valid(int32_t key, uint32_t row, uint32_t qlen, uint32_t klen) {
        return key >= 0 && key < static_cast<int32_t>(klen) &&
               key <= static_cast<int64_t>(row) + klen - qlen;
    }
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
        GM_ADDR weights, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
        GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR actualQ, GM_ADDR actualK,
        GM_ADDR dQuery, GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR outputLoss, GM_ADDR workspace,
        const SparseLightningIndexerGradKlLossTilingData &tiling) {
        t = tiling;
        q.SetGlobalBuffer((__gm__ T*)query); k.SetGlobalBuffer((__gm__ T*)key);
        qi.SetGlobalBuffer((__gm__ T*)queryIndex); ki.SetGlobalBuffer((__gm__ T*)keyIndex);
        w.SetGlobalBuffer((__gm__ W*)weights); indices.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        qr.SetGlobalBuffer((__gm__ QR*)queryRope); kr.SetGlobalBuffer((__gm__ KR*)keyRope);
        maxima.SetGlobalBuffer((__gm__ float*)softmaxMax); sums.SetGlobalBuffer((__gm__ float*)softmaxSum);
        ql32.SetGlobalBuffer((__gm__ int32_t*)actualQ); ql64.SetGlobalBuffer((__gm__ int64_t*)actualQ);
        kl32.SetGlobalBuffer((__gm__ int32_t*)actualK); kl64.SetGlobalBuffer((__gm__ int64_t*)actualK);
        dq.SetGlobalBuffer((__gm__ T*)dQuery); dk.SetGlobalBuffer((__gm__ T*)dKey);
        dw.SetGlobalBuffer((__gm__ W*)dWeight); loss.SetGlobalBuffer((__gm__ float*)outputLoss);
        scratch.SetGlobalBuffer((__gm__ float*)workspace + GetBlockIdx()*t.batchScratch);
        losses.SetGlobalBuffer((__gm__ float*)workspace + t.cores*t.batchScratch);
        stride = (t.indexDim + 7)/8*8;
        uint32_t cacheSize = t.indexHeads*stride;
        pipe.InitBuffer(storage, (2*cacheSize + 512*8 + 128*4 + 64)*sizeof(float));
        auto mem = storage.Get<float>();
        a = mem; b = mem[512]; product = mem[1024]; temporary = mem[1536]; result = mem[2048];
        raw = mem[2560].ReinterpretCast<uint8_t>();
        qiCache = mem[3072]; dqCache = mem[3072+cacheSize];
        keyCache = mem[3072+2*cacheSize]; dkCache = keyCache[128];
        gradTemp = dkCache[128]; weightOut = gradTemp[128];
        simOffset = (static_cast<uint64_t>(t.keyRows)*t.indexDim+15)/16*16;
        massOffset = simOffset + static_cast<uint64_t>(t.sparseSize)*t.indexHeads;
        logitOffset = massOffset + t.sparseSize;
    }
    __aicore__ inline SliPair PairAt(uint64_t offset) {
        return SliPair(scratch.GetValue(offset),scratch.GetValue(offset+1));
    }
    __aicore__ inline void PutPair(uint64_t offset,SliPair value) {
        scratch.SetValue(offset,value.hi); scratch.SetValue(offset+1,value.lo);
    }
    template<class A,class B>
    __aicore__ inline SliPair PairDot(GlobalTensor<A> x,uint64_t xo,GlobalTensor<B> y,uint64_t yo,uint32_t count) {
        SliPair sum(0);
        for(uint32_t d=0;d<count;++d) {
            float av=static_cast<float>(x.GetValue(xo+d)), bv=static_cast<float>(y.GetValue(yo+d));
            if(av!=0 && bv!=0) sum=sum+SliPair(av)*SliPair(bv);
        }
        return sum;
    }
    __aicore__ inline SliPair PairMain(uint32_t batch,uint32_t row,uint32_t head,uint32_t key) {
        uint64_t qo=(static_cast<uint64_t>(batch)*t.queryRows+row)*t.queryHeads+head;
        uint64_t ko=static_cast<uint64_t>(batch)*t.keyRows+key;
        return (PairDot(q,qo*t.queryDim,k,ko*t.queryDim,t.queryDim)+
                PairDot(qr,qo*t.ropeDim,kr,ko*t.ropeDim,t.ropeDim))*SliPair(t.scale);
    }
    __aicore__ inline void ProcessAccurate() {
        uint64_t keyCount=static_cast<uint64_t>(t.keyRows)*t.indexDim;
        uint64_t qOffset=(2*keyCount+15)/16*16;
        uint64_t sOffset=qOffset+2*t.indexHeads*t.indexDim;
        uint64_t mOffset=sOffset+2*static_cast<uint64_t>(t.sparseSize)*t.indexHeads;
        uint64_t iOffset=mOffset+2*t.sparseSize;
        SliPair total(0);
        for(uint32_t batch=GetBlockIdx();batch<t.batch;batch+=t.cores) {
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(scratch);
            Duplicate(a,0.0f,512);
            for(uint64_t offset=0;offset<2*keyCount;offset+=512) {
                uint32_t n=2*keyCount-offset<512?2*keyCount-offset:512;
                Store(scratch,offset,a,n);
            }
            uint32_t qlen=t.empty?0:Length(true,batch),klen=Length(false,batch);
            for(uint32_t row=0;row<t.queryRows;++row) {
                uint64_t token=static_cast<uint64_t>(batch)*t.queryRows+row;
                for(uint32_t d=0;d<t.indexHeads*t.indexDim;++d) PutPair(qOffset+2*d,SliPair(0));
                Duplicate(weightOut,0.0f,64); PipeBarrier<PIPE_ALL>();
                if(row<qlen) {
                    SliPair headNorm[128], headScore[128], gradW[64];
                    for(uint32_t h=0;h<t.indexHeads;++h) gradW[h]=SliPair(0);
                    int64_t visible=static_cast<int64_t>(row)+klen-qlen+1;
                    // FP32 compatibility inputs may carry rounded statistics. Reconstruct
                    // the visible normalizer with the same precision as the sparse scores.
                    for(uint32_t h=0;h<t.queryHeads;++h) {
                        SliPair mx(-3.4e38f),sm(0);
                        for(int64_t key=0;key<visible;++key) {
                            SliPair score=PairMain(batch,row,h,key);
                            if(score>mx) mx=score;
                        }
                        for(int64_t key=0;key<visible;++key)
                            sm=sm+SliPairExp(PairMain(batch,row,h,key)-mx);
                        headNorm[h]=mx+SliPairLog(sm);
                    }
                    uint32_t validCount=0;
                    SliPair maxMass(-3.4e38f),maxLogit(-3.4e38f);
                    for(uint32_t j=0;j<t.sparseSize;++j) {
                        int32_t key=indices.GetValue(token*t.sparseSize+j);
                        if(!Valid(key,row,qlen,klen)) continue;
                        ++validCount;
                        SliPair mh(-3.4e38f),mass(0),logit(0);
                        for(uint32_t h=0;h<t.queryHeads;++h) {
                            headScore[h]=PairMain(batch,row,h,key)-headNorm[h];
                            if(headScore[h]>mh) mh=headScore[h];
                        }
                        for(uint32_t h=0;h<t.queryHeads;++h) mass=mass+SliPairExp(headScore[h]-mh);
                        SliPair logmass=mh+SliPairLog(mass);
                        PutPair(mOffset+2*j,logmass);
                        if(logmass>maxMass) maxMass=logmass;
                        for(uint32_t h=0;h<t.indexHeads;++h) {
                            SliPair similarity=PairDot(qi,(token*t.indexHeads+h)*t.indexDim,
                                ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim,t.indexDim);
                            PutPair(sOffset+2*(static_cast<uint64_t>(j)*t.indexHeads+h),similarity);
                            if(similarity>SliPair(0)) logit=logit+SliPair(static_cast<float>(w.GetValue(token*t.indexHeads+h)))*similarity;
                        }
                        PutPair(iOffset+2*j,logit);
                        if(logit>maxLogit) maxLogit=logit;
                    }
                    if(validCount) {
                        SliPair mt(0),ip(0);
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            mt=mt+SliPairExp(PairAt(mOffset+2*j)-maxMass);
                            ip=ip+SliPairExp(PairAt(iOffset+2*j)-maxLogit);
                        }
                        SliPair logmt=SliPairLog(mt),logip=SliPairLog(ip);
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            SliPair tm=PairAt(mOffset+2*j)-maxMass, im=PairAt(iOffset+2*j)-maxLogit;
                            SliPair target=SliPairExp(tm)/mt,predicted=SliPairExp(im)/ip,di=predicted-target;
                            total=total+target*((tm-logmt)-(im-logip));
                            for(uint32_t h=0;h<t.indexHeads;++h) {
                                SliPair similarity=PairAt(sOffset+2*(static_cast<uint64_t>(j)*t.indexHeads+h));
                                if(!(similarity>SliPair(0))) continue;
                                gradW[h]=gradW[h]+di*similarity;
                                SliPair ds=di*SliPair(static_cast<float>(w.GetValue(token*t.indexHeads+h)));
                                if(ds.hi==0 && ds.lo==0) continue;
                                for(uint32_t d=0;d<t.indexDim;++d) {
                                    uint64_t qoff=qOffset+2*(h*t.indexDim+d),koff=2*(static_cast<uint64_t>(key)*t.indexDim+d);
                                    PutPair(qoff,PairAt(qoff)+ds*SliPair(static_cast<float>(ki.GetValue(
                                        (static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim+d))));
                                    PutPair(koff,PairAt(koff)+ds*SliPair(static_cast<float>(qi.GetValue(
                                        (token*t.indexHeads+h)*t.indexDim+d))));
                                }
                            }
                        }
                        for(uint32_t h=0;h<t.indexHeads;++h) weightOut.SetValue(h,gradW[h].Value());
                    }
                }
                for(uint32_t h=0;h<t.indexHeads;++h) {
                    for(uint32_t d=0;d<t.indexDim;++d) a.SetValue(d,PairAt(qOffset+2*(h*t.indexDim+d)).Value());
                    Store(dq,(token*t.indexHeads+h)*t.indexDim,a,t.indexDim);
                }
                Store(dw,token*t.indexHeads,weightOut,t.indexHeads);
            }
            for(uint64_t offset=0;offset<keyCount;offset+=512) {
                uint32_t n=keyCount-offset<512?keyCount-offset:512;
                for(uint32_t d=0;d<n;++d) a.SetValue(d,PairAt(2*(offset+d)).Value());
                Store(dk,static_cast<uint64_t>(batch)*keyCount+offset,a,n);
            }
        }
        losses.SetValue(GetBlockIdx()*16,total.hi); losses.SetValue(GetBlockIdx()*16+1,total.lo);
        DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(losses);
        if(t.cores>1) SyncAll();
        if(GetBlockIdx()==0) {
            SliPair sum(0);
            for(uint32_t i=0;i<t.cores;++i) sum=sum+SliPair(losses.GetValue(i*16),losses.GetValue(i*16+1));
            loss.SetValue(0,sum.Value());
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(loss);
        }
    }
    __aicore__ inline void Process() {
        if constexpr (AscendC::IsSameType<T,float>::value) { ProcessAccurate(); return; }
        float totalLoss = 0, lossCorrection = 0;
        for (uint32_t batch = GetBlockIdx(); batch < t.batch; batch += t.cores) {
            uint64_t keyCount = static_cast<uint64_t>(t.keyRows)*t.indexDim;
            Duplicate(a, 0.0f, 512);
            for (uint64_t offset = 0; offset < keyCount; offset += 512) {
                uint32_t n = keyCount-offset < 512 ? keyCount-offset : 512;
                Store(scratch, offset, a, n);
            }
            uint32_t qlen = t.empty ? 0 : Length(true,batch), klen = Length(false,batch);
            for (uint32_t row = 0; row < t.queryRows; ++row) {
                Duplicate(dqCache, 0.0f, t.indexHeads*stride);
                Duplicate(weightOut, 0.0f, 64);
                PipeBarrier<PIPE_ALL>();
                uint64_t token = static_cast<uint64_t>(batch)*t.queryRows+row;
                if (row < qlen) {
                    for (uint32_t h=0; h<t.indexHeads; ++h)
                        Load(qiCache[h*stride], qi, (token*t.indexHeads+h)*t.indexDim, t.indexDim);
                    float headMax[128], headSum[128], score[128], weight[64], dwSum[64], dwCorr[64];
                    for (uint32_t h=0; h<t.indexHeads; ++h) {
                        if constexpr (AscendC::IsSameType<W,bfloat16_t>::value)
                            weight[h] = AscendC::ToFloat(w.GetValue(token*t.indexHeads+h));
                        else weight[h] = static_cast<float>(w.GetValue(token*t.indexHeads+h));
                        dwSum[h]=0; dwCorr[h]=0;
                    }
                    for (uint32_t h=0; h<t.queryHeads; ++h) {
                        uint64_t so = static_cast<uint64_t>(batch)*t.queryRows*t.queryHeads +
                            (t.statsLayout ? h*t.queryRows+row : row*t.queryHeads+h);
                        headMax[h] = maxima.GetValue(so); headSum[h] = sums.GetValue(so);
                        if constexpr (AscendC::IsSameType<T,float>::value) {
                            float mx = -3.4e38f;
                            int64_t visible = static_cast<int64_t>(row)+klen-qlen+1;
                            for (int64_t key=0; key<visible; ++key) {
                                float v=MainScore(batch,row,h,key); if(v>mx) mx=v;
                            }
                            float sm=0, correction=0;
                            for (int64_t key=0; key<visible; ++key)
                                SliSum(SliExp(MainScore(batch,row,h,key)-mx),sm,correction);
                            headMax[h]=mx; headSum[h]=sm;
                        }
                    }
                    float maxMass=-3.4e38f, maxLogit=-3.4e38f;
                    uint32_t validCount=0;
                    for (uint32_t j=0; j<t.sparseSize; ++j) {
                        int32_t key=indices.GetValue(token*t.sparseSize+j);
                        if (!Valid(key,row,qlen,klen)) continue;
                        ++validCount;
                        float mh=-3.4e38f;
                        for(uint32_t h=0;h<t.queryHeads;++h) {
                            score[h]=MainScore(batch,row,h,key)-headMax[h]-SliLog(headSum[h]);
                            if(score[h]>mh) mh=score[h];
                        }
                        float mass=0, mc=0;
                        for(uint32_t h=0;h<t.queryHeads;++h) SliSum(SliExp(score[h]-mh),mass,mc);
                        float logmass=mh+SliLog(mass);
                        scratch.SetValue(massOffset+j,logmass);
                        if(logmass>maxMass) maxMass=logmass;
                        Load(keyCache,ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim,t.indexDim);
                        float logit=0, lc=0;
                        for(uint32_t h=0;h<t.indexHeads;++h) {
                            Mul(product,qiCache[h*stride],keyCache,t.indexDim);
                            PipeBarrier<PIPE_V>();
                            ReduceSum(result,product,temporary,t.indexDim);
                            PipeBarrier<PIPE_ALL>();
                            float similarity=result.GetValue(0);
                            scratch.SetValue(simOffset+static_cast<uint64_t>(j)*t.indexHeads+h,similarity);
                            SliSum(weight[h]*(similarity>0?similarity:0),logit,lc);
                        }
                        scratch.SetValue(logitOffset+j,logit);
                        if(logit>maxLogit) maxLogit=logit;
                    }
                    if(validCount) {
                        float mt=0, mtc=0, ip=0, ipc=0;
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            SliSum(SliExp(scratch.GetValue(massOffset+j)-maxMass),mt,mtc);
                            SliSum(SliExp(scratch.GetValue(logitOffset+j)-maxLogit),ip,ipc);
                        }
                        float logmt=SliLog(mt), logip=SliLog(ip);
                        for(uint32_t j=0;j<t.sparseSize;++j) {
                            int32_t key=indices.GetValue(token*t.sparseSize+j);
                            if(!Valid(key,row,qlen,klen)) continue;
                            float lt=(scratch.GetValue(massOffset+j)-maxMass)-logmt;
                            float lp=(scratch.GetValue(logitOffset+j)-maxLogit)-logip;
                            float target=SliExp(lt), predicted=SliExp(lp), di=predicted-target;
                            SliSum(target*(lt-lp),totalLoss,lossCorrection);
                            Load(keyCache,ki,(static_cast<uint64_t>(batch)*t.keyRows+key)*t.indexDim,t.indexDim);
                            Load(dkCache,scratch,static_cast<uint64_t>(key)*t.indexDim,t.indexDim);
                            for(uint32_t h=0;h<t.indexHeads;++h) {
                                float similarity=scratch.GetValue(simOffset+static_cast<uint64_t>(j)*t.indexHeads+h);
                                if(similarity<=0) continue;
                                SliSum(di*similarity,dwSum[h],dwCorr[h]);
                                float ds=di*weight[h];
                                Muls(gradTemp,keyCache,ds,t.indexDim); PipeBarrier<PIPE_V>();
                                Add(dqCache[h*stride],dqCache[h*stride],gradTemp,t.indexDim); PipeBarrier<PIPE_V>();
                                Muls(gradTemp,qiCache[h*stride],ds,t.indexDim); PipeBarrier<PIPE_V>();
                                Add(dkCache,dkCache,gradTemp,t.indexDim); PipeBarrier<PIPE_V>();
                            }
                            Store(scratch,static_cast<uint64_t>(key)*t.indexDim,dkCache,t.indexDim);
                        }
                        for(uint32_t h=0;h<t.indexHeads;++h) weightOut.SetValue(h,dwSum[h]);
                    }
                }
                for(uint32_t h=0;h<t.indexHeads;++h)
                    Store(dq,(token*t.indexHeads+h)*t.indexDim,dqCache[h*stride],t.indexDim);
                Store(dw,token*t.indexHeads,weightOut,t.indexHeads);
            }
            for(uint64_t offset=0;offset<keyCount;offset+=512) {
                uint32_t n=keyCount-offset<512?keyCount-offset:512;
                Load(a,scratch,offset,n);
                Store(dk,static_cast<uint64_t>(batch)*keyCount+offset,a,n);
            }
        }
        losses.SetValue(GetBlockIdx()*16,totalLoss);
        DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(losses);
        if (t.cores > 1) SyncAll();
        if(GetBlockIdx()==0) {
            float sum=0, correction=0;
            for(uint32_t i=0;i<t.cores;++i) SliSum(losses.GetValue(i*16),sum,correction);
            loss.SetValue(0,sum);
            DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE>(loss);
        }
    }
};

template<typename DT_QUERY, typename DT_WEIGHT, typename DT_QROPE, typename DT_KROPE>
__global__ __aicore__ void sparse_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weights,
    GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR queryRope,
    GM_ADDR keyRope, GM_ADDR actualQ, GM_ADDR actualK, GM_ADDR dQuery, GM_ADDR dKey,
    GM_ADDR dWeight, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(SparseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseLightningIndexerGradKlLossTilingData,t,tiling);
    SparseGrad<DT_QUERY,DT_WEIGHT,DT_QROPE,DT_KROPE> op;
    op.Init(query,key,queryIndex,keyIndex,weights,sparseIndices,softmaxMax,softmaxSum,
        queryRope,keyRope,actualQ,actualK,dQuery,dKey,dWeight,loss,GetUserWorkspace(workspace),t);
    op.Process();
}
