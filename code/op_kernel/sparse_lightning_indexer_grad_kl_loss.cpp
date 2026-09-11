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
// The tiny path runs on the vector-only launch, which carries no hardware
// barrier, so the cores rendezvous through the workspace: every producer stamps
// its ready word with a value derived from the launch epoch and the row owner
// folds the partials once every stamp of this launch is visible.  Mixing the
// epoch keeps a stamp from a previous launch from ever matching.
__aicore__ inline uint32_t SliStamp(uint32_t epoch, uint32_t core) {
    uint32_t v = (epoch + 1u) * 2654435761u + core * 40503u + 0x165667B1u;
    v ^= v >> 15; v *= 2246822519u; v ^= v >> 13;
    return v;
}
// Ready words sit one per core at the end of its workspace slot, the launch epoch
// in the reserved sync span behind the slots.  Both travel through the DMA path,
// which needs no data-cache maintenance between cores.
constexpr uint32_t kEpochLane = 0u;
// each 4-byte stamp occupies its own 32B block in the polling buffer
constexpr uint32_t kStampStride = 8u;
// the cleared-output stamp sits behind the launch epoch in the sync span
constexpr uint32_t kClearLane = 8u;
// the cleared-output stamp lives in its own namespace
constexpr uint32_t kClearStamp = 0x2000u;
// spare lanes of the per-core auxiliary block: the epoch word, the poll buffer and
// the loss share staged for the FP32 atomic fold
constexpr uint32_t kEpochLaneOffset = 832u;
constexpr uint32_t kLossLane = 960u;
constexpr uint32_t kStampLaneOffset = 896u;

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
    if(x.hi==0 && x.lo==0) return SliPair(1);
    float z=x.hi*1.4426950408889634f;
    int n=static_cast<int>(z+(z>=0?0.5f:-0.5f));
    SliPair r=x-SliPair(0.693147182464599609375f,-1.904654323148236e-9f)*SliPair(static_cast<float>(n));
    // Centering the interval bounds the degree-12 Taylor remainder below 3e-16.
    SliPair sum=SliPair(2.08767558795841524e-09f,1.10828398092043424e-16f);
    sum=sum*r+SliPair(2.50521079436794025e-08f,4.41762304464836646e-16f);
    sum=sum*r+SliPair(2.75573199814971304e-07f,-7.57511220905119492e-15f);
    sum=sum*r+SliPair(2.75573188446287531e-06f,3.79357122429722915e-14f);
    sum=sum*r+SliPair(2.48015876422869042e-05f,-3.40699609366681977e-13f);
    sum=sum*r+SliPair(1.98412701138295233e-04f,-2.72559687493345582e-12f);
    sum=sum*r+SliPair(1.38888892251998186e-03f,-3.36310944371032150e-11f);
    sum=sum*r+SliPair(8.33333376795053482e-03f,-4.34617203337595015e-10f);
    sum=sum*r+SliPair(4.16666679084300995e-02f,-1.24176346982807218e-09f);
    sum=sum*r+SliPair(1.66666671633720398e-01f,-4.96705387931228870e-09f);
    sum=sum*r+SliPair(5.00000000000000000e-01f,0.00000000000000000e+00f);
    sum=sum*r+SliPair(1.00000000000000000e+00f,0.00000000000000000e+00f);
    sum=sum*r+SliPair(1.00000000000000000e+00f,0.00000000000000000e+00f);
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
    TPipe *pipe;
    TBuf<TPosition::VECCALC> storage;
    LocalTensor<float> a, b, product, temporary, result, qiCache, dqCache, keyCache, dkCache, gradTemp, weightOut;
    LocalTensor<uint8_t> raw;
    uint32_t stride;
    uint64_t simOffset, massOffset, logitOffset;

    // ---- tiny fast path layout (all offsets in float elements) ----
    uint32_t fN1, fNI, fDq, fDr, fDi, fK, fKP, fS2, fRows;
    uint64_t fOffQ, fOffQR, fOffQI, fOffW, fOffMax, fOffSum, fOffC;
    uint64_t fOffKey, fOffKeyR, fOffKi, fOffSim, fOffSelK, fOffSelKR, fOffSelKI;
    uint64_t fOffDk, fOffDq, fOffDw, fOffProd, fOffAux, fOffIdx, fOffRed;
    uint64_t fOffDkProd, fOffDsSpr, fOffDiSpr, fOffHeadScr;
    GlobalTensor<float> wsAll;
    GlobalTensor<uint32_t> stamps;
    GlobalTensor<uint32_t> epochWord;
    GlobalTensor<uint32_t> clearWord;
    uint32_t launchEpoch;
    uint64_t fOffSco, fOffPos, fOffDiMat, fOffV, fOffU, fOffUmat, fOffDqAcc, fOffWStage;
    uint32_t fWStageStride;
    uint32_t fN1P, fNIP;
    uint64_t fStgQ, fStgQR, fStgQI, fStgW, fStgKey, fStgKeyR, fStgKi;
    uint32_t fStgQElems;

    __aicore__ inline uint64_t FastTake(uint64_t &cursor, uint64_t count) {
        cursor = (cursor + 7) / 8 * 8;
        uint64_t at = cursor;
        cursor += count + 64;                 // 32B alignment plus read-window slack
        return at;
    }

    __aicore__ inline uint64_t FastLayout() {
        fN1 = t.queryHeads; fNI = t.indexHeads;
        fDq = t.queryDim; fDr = t.ropeDim; fDi = t.indexDim;
        fK = t.sparseSize; fKP = (fK + 7) / 8 * 8;
        fS2 = t.keyRows; fRows = t.batch * t.queryRows;
        uint64_t rows = fRows, n1 = fN1, ni = fNI, dq = fDq, dr = fDr, di = fDi;
        uint64_t batches = t.batch, s2 = fS2, kp = fKP, k = fK;
        uint64_t o = 0;
        fOffQ = FastTake(o, rows*n1*dq);
        fOffQR = FastTake(o, rows*n1*dr);
        fOffQI = FastTake(o, rows*ni*di);
        uint64_t n1p0 = (n1 + 7) / 8 * 8, nip0 = (ni + 7) / 8 * 8;
        uint64_t wStage = (ni + 15) / 16 * 16;               // 32B-aligned 16-bit staging rows
        fOffW = FastTake(o, rows*nip0);
        fOffMax = FastTake(o, rows*n1p0);
        fOffSum = FastTake(o, rows*n1p0);
        fOffC = FastTake(o, rows*n1p0);
        fOffWStage = FastTake(o, rows*wStage);
        fOffKey = FastTake(o, batches*s2*dq);
        fOffKeyR = FastTake(o, batches*s2*dr);
        fOffKi = FastTake(o, batches*s2*di);
        uint64_t n1p = n1p0, nip = nip0;
        fN1P = static_cast<uint32_t>(n1p); fNIP = static_cast<uint32_t>(nip);
        fWStageStride = static_cast<uint32_t>(wStage);
        fOffSco = FastTake(o, kp*n1p);
        fOffSim = FastTake(o, kp*nip);
        fOffPos = FastTake(o, kp*nip);
        fOffDiMat = FastTake(o, kp*nip);
        if (t.tiny) {
            fOffV = 0; fOffU = 0; fOffUmat = 0; fOffDqAcc = 0;
            fOffSelK = 0; fOffSelKR = 0; fOffSelKI = 0;
        } else {
            fOffV = FastTake(o, kp*nip);
            fOffU = FastTake(o, kp*nip);
            fOffUmat = FastTake(o, nip*di);
            fOffDqAcc = FastTake(o, nip*di);
            fOffSelK = FastTake(o, kp*dq);
            fOffSelKR = FastTake(o, kp*dr);
            fOffSelKI = FastTake(o, kp*di);
        }
        fOffDk = FastTake(o, batches*s2*di);
        fOffDq = FastTake(o, rows*ni*di);
        fOffDw = FastTake(o, rows*nip0);
        uint64_t redSpan = (uint64_t)batches*s2*di;
        fOffRed = FastTake(o, t.tiny ? (t.cores + 1)*(redSpan + 8) + 16
                                     : redSpan + 3*((redSpan + 15)/16*16) + 16);
        uint64_t wide = (s2 > kp ? s2 : kp);
        uint64_t dim = dq > dr ? dq : dr;
        if (di > dim) dim = di;
        uint64_t prodNeed = wide*dim;                        // visible-window products
        uint64_t split = n1p*dq + 64 + nip*di;               // per-key products + head tree
        if (split > prodNeed) prodNeed = split;
        if (kp*nip > prodNeed) prodNeed = kp*nip;
        fOffProd = FastTake(o, prodNeed);
        fOffAux = FastTake(o, 2048);
        fOffIdx = FastTake(o, rows*k);
        // tiny path scratch: per-key indexer products and the block-spread copies
        // that let one iteration carry a whole vector of per-key constants
        if (t.tiny) {
            fOffDkProd = 0;
            fOffDsSpr = FastTake(o, 8*kp*nip);
            fOffDiSpr = FastTake(o, 8*kp);
            fOffHeadScr = FastTake(o, n1p*((s2 + 7) / 8 * 8));
        } else {
            fOffDkProd = 0; fOffDsSpr = 0; fOffDiSpr = 0; fOffHeadScr = 0;
        }
        // One staging slice per input tensor lets every transfer be issued before a
        // single wait; slices only need 32B alignment.
        // One staging slice per non-float32 input tensor lets every transfer be
        // issued before a single wait; slices only need 32B alignment.
        bool stageT = !AscendC::IsSameType<T,float>::value;
        bool stageW = !AscendC::IsSameType<W,float>::value;
        bool stageQR = !AscendC::IsSameType<QR,float>::value;
        bool stageKR = !AscendC::IsSameType<KR,float>::value;
        uint64_t stg = (o + 7) / 8 * 8;
        fStgQElems = static_cast<uint32_t>(rows*n1*dq);
        fStgQ = stg;     if (stageT) stg = (stg + rows*n1*dq + 7) / 8 * 8;
        fStgQR = stg;    if (stageQR) stg = (stg + rows*n1*dr + 7) / 8 * 8;
        fStgQI = stg;    if (stageT) stg = (stg + rows*ni*di + 7) / 8 * 8;
        fStgW = stg;     if (stageW) stg = (stg + rows*ni + 7) / 8 * 8;
        fStgKey = stg;   if (stageT) stg = (stg + batches*s2*dq + 7) / 8 * 8;
        fStgKeyR = stg;  if (stageKR) stg = (stg + batches*s2*dr + 7) / 8 * 8;
        fStgKi = stg;    if (stageT) stg = (stg + batches*s2*di + 7) / 8 * 8;
        return stg + 64;
    }

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
    // The tiny path stages the length arrays once per core; a scalar global load
    // per batch would otherwise stall the scalar pipe on every row.
    __aicore__ inline uint32_t CachedLength(bool query, uint32_t batch, LocalTensor<int32_t> cache) {
        uint32_t type = query ? t.queryLengthType : t.keyLengthType;
        uint32_t bound = query ? t.queryRows : t.keyRows;
        if (!type) return bound;
        int64_t value = type == 1 ? static_cast<int64_t>(cache.GetValue(batch))
                                  : cache.ReinterpretCast<int64_t>().GetValue(batch);
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
        TPipe *pipeIn, const SparseLightningIndexerGradKlLossTilingData &tiling) {
        t = tiling;
        pipe = pipeIn;
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
        wsAll.SetGlobalBuffer((__gm__ float*)workspace);
        losses.SetGlobalBuffer((__gm__ float*)workspace + t.cores*t.batchScratch);
        uint64_t syncBase = (uint64_t)t.cores*t.batchScratch + (uint64_t)t.cores*16;
        epochWord.SetGlobalBuffer((__gm__ uint32_t*)workspace + syncBase);
        clearWord.SetGlobalBuffer((__gm__ uint32_t*)workspace + syncBase + kClearLane);
        stamps.SetGlobalBuffer((__gm__ uint32_t*)workspace +
            (uint64_t)t.batch*t.keyRows*t.indexDim + 1);
        stride = (t.indexDim + 7)/8*8;
        uint32_t cacheSize = t.indexHeads*stride;
        uint64_t genFloats = 2*(uint64_t)cacheSize + 512*8 + 128*4 + 64;
        uint64_t fastFloats = t.fast ? FastLayout() : 0;
        uint64_t totalFloats = genFloats > fastFloats ? genFloats : fastFloats;
        pipe->InitBuffer(storage, totalFloats*sizeof(float));
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
    // The row owner folds every producer's partial in one burst: the accumulator
    // slab holds one slot per participating core, so the fold is a log-depth tree
    // of adds over lanes that are already in registers.
    __aicore__ inline LocalTensor<float> MergePartials(LocalTensor<float> base, uint32_t cores,
                                                       uint32_t span, uint32_t slot, uint32_t tail) {
        LocalTensor<float> sum = base[fOffRed];
        // the accumulator is one lane longer than a partial, so the read buffer has
        // to start past it or the folded loss lane would overlap the first partial
        LocalTensor<float> buf = base[fOffRed][span + 8];
        SyncVToMte2();
        DataCopyExtParams pr{1, static_cast<uint32_t>(cores*slot*sizeof(float)), 0, 0, 0};
        DataCopyPad(buf, wsAll[0], pr, DataCopyPadExtParams<float>{false,0,0,0});
        SyncMte2ToV();
        uint32_t left = cores;
        while (left > 1) {
            uint32_t half = left / 2;
            Add(buf, buf, buf[(uint64_t)half*slot], half*slot);
            if (left & 1u) {
                Adds(buf[(uint64_t)half*slot], buf[(uint64_t)(left - 1)*slot], 0.0f, slot);
                left = half + 1;
            } else {
                left = half;
            }
        }
        Adds(sum, buf, 0.0f, span + tail);
        return sum;
    }
    __aicore__ inline void SyncMte2ToS() {
        TEventID id = pipe->FetchEventID(HardEvent::MTE2_S);
        SetFlag<HardEvent::MTE2_S>(id);
        WaitFlag<HardEvent::MTE2_S>(id);
    }
    __aicore__ inline void SyncMte2ToV() {
        TEventID id = pipe->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }
    __aicore__ inline void SyncVToMte3() {
        TEventID id = pipe->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }
    __aicore__ inline void SyncVToMte2() {
        TEventID id = pipe->FetchEventID(HardEvent::V_MTE2);
        SetFlag<HardEvent::V_MTE2>(id);
        WaitFlag<HardEvent::V_MTE2>(id);
    }
    __aicore__ inline void SyncMte3ToV() {
        TEventID id = pipe->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(id);
        WaitFlag<HardEvent::MTE3_V>(id);
    }
    __aicore__ inline void SyncVToS() {
        TEventID id = pipe->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(id);
        WaitFlag<HardEvent::V_S>(id);
    }
    __aicore__ inline void SyncSToV() {
        TEventID id = pipe->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(id);
        WaitFlag<HardEvent::S_V>(id);
    }
    __aicore__ inline void SyncSToMte3() {
        TEventID id = pipe->FetchEventID(HardEvent::S_MTE3);
        SetFlag<HardEvent::S_MTE3>(id);
        WaitFlag<HardEvent::S_MTE3>(id);
    }
    template<class U>
    __aicore__ inline void FastPad(LocalTensor<U> dst, GlobalTensor<U> src, uint64_t offset, uint32_t count) {
        uint32_t done = 0;
        while (done < count) {
            uint32_t chunk = count - done;
            if (chunk > 16384) chunk = 16384;
            DataCopyExtParams p{1, static_cast<uint32_t>(chunk*sizeof(U)), 0, 0, 0};
            // a short block is padded with zeros, which is what the narrow rows need
            DataCopyPad(dst[done], src[offset+done], p, DataCopyPadExtParams<U>{true, 0, 0, static_cast<U>(0)});
            done += chunk;
        }
    }
    template<class U>
    __aicore__ inline void FastDma(LocalTensor<float> dst, GlobalTensor<U> src, uint64_t offset,
                                   uint32_t count, uint64_t stage) {
        if constexpr (AscendC::IsSameType<U,float>::value) {
            FastPad(dst, src, offset, count);
        } else {
            FastPad(a[stage].ReinterpretCast<U>(), src, offset, count);
        }
    }
    template<class U>
    __aicore__ inline void FastCast(LocalTensor<float> dst, uint64_t stage, uint32_t count) {
        if constexpr (!AscendC::IsSameType<U,float>::value) {
            Cast(dst, a[stage].ReinterpretCast<U>(), RoundMode::CAST_NONE, static_cast<int32_t>(count));
        }
    }
    template<class U>
    __aicore__ inline void FastStore(GlobalTensor<U> dst, uint64_t offset, LocalTensor<float> src, uint32_t count) {
        if (count == 0) return;
        if constexpr (AscendC::IsSameType<U,float>::value) {
            SyncVToMte3();
            DataCopyExtParams p{1, static_cast<uint32_t>(count*sizeof(float)), 0, 0, 0};
            DataCopyPad(dst[offset], src, p);
        } else {
            // the input staging area is dead by now; reuse it to narrow the outputs
            uint32_t done = 0;
            auto stage = a[fStgQ].ReinterpretCast<U>();
            while (done < count) {
                uint32_t chunk = count - done;
                if (chunk > fStgQElems) chunk = fStgQElems;
                SyncMte3ToV();   // the staging slot must be free before rewriting it
                Cast(stage, src[done], RoundMode::CAST_RINT, static_cast<int32_t>(chunk));
                SyncVToMte3();
                DataCopyExtParams p{1, static_cast<uint32_t>(chunk*sizeof(U)), 0, 0, 0};
                DataCopyPad(dst[offset+done], stage, p);
                done += chunk;
            }
        }
    }

    // Publish the query gradient and the weight gradient.  When both carry the core
    // dtype they share one staging pass, so the pipeline drains once for the pair
    // instead of once per output; padded weight rows are narrowed on the way out.
    __aicore__ inline void StoreRowGradients(uint64_t first, uint32_t mine) {
        if (mine == 0) return;
        uint32_t qCount = mine*fNI*fDi;
        uint32_t wCount = mine*fNIP;
        if constexpr (AscendC::IsSameType<T,W>::value) {
            if (qCount + wCount <= fStgQElems && (mine <= 1 || fNIP % 16 == 0)) {
                if constexpr (AscendC::IsSameType<T,float>::value) {
                    SyncVToMte3();
                    DataCopyExtParams pq{1, static_cast<uint32_t>(qCount*sizeof(float)), 0, 0, 0};
                    DataCopyPad(dq[first*fNI*fDi], a[fOffDq], pq);
                    for (uint32_t r = 0; r < mine; ++r) {
                        DataCopyExtParams pw{1, static_cast<uint32_t>(fNI*sizeof(float)), 0, 0, 0};
                        DataCopyPad(dw[(first + r)*fNI], a[fOffDw][(uint64_t)r*fNIP], pw);
                    }
                } else {
                    auto stage = a[fStgQ].ReinterpretCast<T>();
                    SyncMte3ToV();
                    Cast(stage, a[fOffDq], RoundMode::CAST_RINT, static_cast<int32_t>(qCount));
                    Cast(stage[qCount], a[fOffDw], RoundMode::CAST_RINT, static_cast<int32_t>(wCount));
                    SyncVToMte3();
                    DataCopyExtParams pq{1, static_cast<uint32_t>(qCount*sizeof(T)), 0, 0, 0};
                    DataCopyPad(dq[first*fNI*fDi], stage, pq);
                    for (uint32_t r = 0; r < mine; ++r) {
                        DataCopyExtParams pw{1, static_cast<uint32_t>(fNI*sizeof(T)), 0, 0, 0};
                        DataCopyPad(dw[(first + r)*fNI], stage[qCount + (uint64_t)r*fNIP], pw);
                    }
                }
                return;
            }
        }
        FastStore(dq, first*fNI*fDi, a[fOffDq], qCount);
        for (uint32_t r = 0; r < mine; ++r)
            FastStore(dw, (first + r)*fNI, a[fOffDw][(uint64_t)r*fNIP], fNI);
    }

    // Sum the rows of a (rows x width) matrix into row 0; rows must be a power of two.
    __aicore__ inline LocalTensor<float> SumRows(LocalTensor<float> a0, LocalTensor<float> a1,
                                                 uint32_t rows, uint32_t width) {
        LocalTensor<float> cur = a0, nxt = a1;
        bool atFirst = true;
        uint32_t left = rows;
        while (left > 1) {
            uint32_t half = left / 2;
            Add(nxt, cur, cur[width], width, (uint8_t)half,
                BinaryRepeatParams(1, 1, 1, width/8, 2*width/8, 2*width/8));
            cur = nxt;
            nxt = atFirst ? a1 : a0;
            atFirst = !atFirst;
            left = half;
        }
        return cur;
    }

    // Single block, all inputs resident in UB, whole tensors moved in one burst.
    // Score work is organized per selected key so one vector op covers every head.
    __aicore__ inline void ProcessFast() {
        LocalTensor<float> base = a;
        LocalTensor<int32_t> idxT = base[fOffIdx].ReinterpretCast<int32_t>();
        uint32_t keys = static_cast<uint32_t>((uint64_t)t.batch*fS2);
        // (batch, row) tokens are split evenly across the launched cores.
        uint32_t cores = t.cores ? t.cores : 1u;
        uint32_t first = GetBlockIdx() * ((fRows + cores - 1) / cores);
        uint32_t last = first + ((fRows + cores - 1) / cores);
        if (last > fRows) last = fRows;
        if (first > fRows) first = fRows;
        uint32_t mine = last - first;
        // narrow rows arrive zero padded from their transfers
        // every transfer is issued before a single pipeline wait
        FastDma(base[fOffQ], q, (uint64_t)first*fN1*fDq, mine*fN1*fDq, fStgQ);
        FastDma(base[fOffQR], qr, (uint64_t)first*fN1*fDr, mine*fN1*fDr, fStgQR);
        FastDma(base[fOffQI], qi, (uint64_t)first*fNI*fDi, mine*fNI*fDi, fStgQI);
        FastDma(base[fOffKey], k, 0, keys*fDq, fStgKey);
        FastDma(base[fOffKeyR], kr, 0, keys*fDr, fStgKeyR);
        FastDma(base[fOffKi], ki, 0, keys*fDi, fStgKi);
        if (mine*fK) FastPad(idxT, indices, (uint64_t)first*fK, mine*fK);
        // narrow rows must start on a 32B boundary: with a block aligned width a
        // single transfer per tensor is enough, otherwise go row by row
        bool narrowAligned = (fNIP == fNI) && (fN1P == fN1);
        if (narrowAligned) {
            if constexpr (AscendC::IsSameType<W,float>::value) {
                FastPad(base[fOffW], w, (uint64_t)first*fNI, mine*fNI);
            } else {
                FastPad(a[fOffWStage].ReinterpretCast<W>(), w, (uint64_t)first*fNI, mine*fNI);
            }
            FastPad(base[fOffMax], maxima, (uint64_t)first*fN1, mine*fN1);
            FastPad(base[fOffSum], sums, (uint64_t)first*fN1, mine*fN1);
        } else {
            for (uint32_t r = 0; r < mine; ++r) {
                uint32_t tr = first + r;
                if constexpr (AscendC::IsSameType<W,float>::value) {
                    FastPad(base[fOffW][r*fNIP], w, (uint64_t)tr*fNI, fNI);
                } else {
                    FastPad(a[fOffWStage].ReinterpretCast<W>()[r*fWStageStride], w, (uint64_t)tr*fNI, fNI);
                }
                FastPad(base[fOffMax][r*fN1P], maxima, (uint64_t)tr*fN1, fN1);
                FastPad(base[fOffSum][r*fN1P], sums, (uint64_t)tr*fN1, fN1);
            }
        }
        SyncMte2ToV();
        FastCast<T>(base[fOffQ], fStgQ, mine*fN1*fDq);
        FastCast<QR>(base[fOffQR], fStgQR, mine*fN1*fDr);
        FastCast<T>(base[fOffQI], fStgQI, mine*fNI*fDi);
        if constexpr (!AscendC::IsSameType<W,float>::value) {
            if (narrowAligned) {
                Cast(base[fOffW], a[fOffWStage].ReinterpretCast<W>(), RoundMode::CAST_NONE,
                     static_cast<int32_t>(mine*fNI));
            } else {
                for (uint32_t r = 0; r < mine; ++r)
                    Cast(base[fOffW][r*fNIP], a[fOffWStage].ReinterpretCast<W>()[r*fWStageStride],
                         RoundMode::CAST_NONE, static_cast<int32_t>(fNI));
            }
        }
        FastCast<T>(base[fOffKey], fStgKey, keys*fDq);
        FastCast<KR>(base[fOffKeyR], fStgKeyR, keys*fDr);
        FastCast<T>(base[fOffKi], fStgKi, keys*fDi);
        Muls(base[fOffQ], base[fOffQ], t.scale, mine*fN1*fDq);
        Muls(base[fOffQR], base[fOffQR], t.scale, mine*fN1*fDr);
        if (!t.fastRecompute) {
            Ln(base[fOffC], base[fOffSum], mine*fN1P);
            Add(base[fOffC], base[fOffC], base[fOffMax], mine*fN1P);
        }
        Duplicate(base[fOffDq], 0.0f, mine*fNI*fDi);
        Duplicate(base[fOffDw], 0.0f, mine*fNIP);

        LocalTensor<float> lossAcc = base[fOffAux][1024];
        Duplicate(lossAcc, 0.0f, 8);
        LocalTensor<float> cVec = base[fOffAux];
        LocalTensor<float> wVec = base[fOffAux][128];
        LocalTensor<float> mass = base[fOffAux][256];
        LocalTensor<float> logit = base[fOffAux][256 + fKP];
        LocalTensor<float> di = base[fOffAux][256 + 2*fKP];
        LocalTensor<float> dwVec = base[fOffAux][256 + 3*fKP];
        LocalTensor<float> work = base[fOffAux][256 + 4*fKP];
        LocalTensor<float> sco = base[fOffSco];
        LocalTensor<float> sim = base[fOffSim];
        LocalTensor<float> pos = base[fOffPos];
        LocalTensor<float> diMat = base[fOffDiMat];
        LocalTensor<float> vMat = base[fOffV];
        LocalTensor<float> uMat = base[fOffU];
        LocalTensor<float> ubScratch = base[fOffUmat];
        LocalTensor<float> dqAcc = base[fOffDqAcc];
        LocalTensor<float> prod = base[fOffProd];
        LocalTensor<float> prod2 = base[fOffProd][(uint64_t)fN1P*fDq + 64];
        LocalTensor<float> selK = base[fOffSelK];
        LocalTensor<float> selKR = base[fOffSelKR];
        LocalTensor<float> selKI = base[fOffSelKI];
        uint32_t sel[64];
        float totalLoss = 0.0f;
        // The supplied statistics may come from a lower-precision forward pass and
        // then no longer reproduce the exact visible normalizer, so the FP32 path
        // rebuilds every head's normalizer over the visible window instead.
        bool useProvided = !t.fastRecompute;
        Duplicate(base[fOffDk], 0.0f, (uint64_t)t.batch*fS2*fDi);
        if (mine) Duplicate(base[fOffDq], 0.0f, mine*fNI*fDi);
        Duplicate(base[fOffDw], 0.0f, fRows*fNIP);
        for (uint32_t b = 0; b < t.batch; ++b) {
            uint32_t qlen = t.empty ? 0 : Length(true, b);
            uint32_t klen = Length(false, b);
            for (uint32_t row = 0; row < t.queryRows; ++row) {
                uint64_t token = (uint64_t)b*t.queryRows + row;
                if (token < first || token >= last) continue;
                if (row >= qlen) continue;
                uint32_t local = (uint32_t)(token - first);
                int32_t visible = static_cast<int32_t>(row) + static_cast<int32_t>(klen)
                                  - static_cast<int32_t>(qlen) + 1;
                uint32_t m = 0;
                for (uint32_t j = 0; j < fK; ++j) {
                    int32_t key = idxT.GetValue((uint64_t)local*fK + j);
                    if (key >= 0 && key < static_cast<int32_t>(klen) && key < visible)
                        sel[m++] = static_cast<uint32_t>(key);
                }
                if (m == 0) continue;
                LocalTensor<float> qRow = base[fOffQ][(uint64_t)local*fN1*fDq];
                LocalTensor<float> qrRow = base[fOffQR][(uint64_t)local*fN1*fDr];
                LocalTensor<float> qiRow = base[fOffQI][(uint64_t)local*fNI*fDi];
                LocalTensor<float> dqRow = base[fOffDq][(uint64_t)local*fNI*fDi];
                LocalTensor<float> dwRow = base[fOffDw][(uint64_t)local*fNIP];
                LocalTensor<float> wRow = base[fOffW][(uint64_t)local*fNIP];
                LocalTensor<float> cRow = base[fOffC][(uint64_t)local*fN1P];
                // ---- target mass over the selected keys ----
                if (!useProvided) {
                    // rebuild per-head log-normalizers over the visible window
                    LocalTensor<float> keyB = base[fOffKey][(uint64_t)b*fS2*fDq];
                    LocalTensor<float> keyRB = base[fOffKeyR][(uint64_t)b*fS2*fDr];
                    uint32_t s2p = (fS2 + 7) / 8 * 8;
                    if (fN1P == fN1 && (uint64_t)fN1P*s2p <= (uint64_t)fKP*fDq) {
                        // Stage every head's window scores in one matrix so a single
                        // reduction replaces one reduction per head.
                        LocalTensor<float> hSc = base[fOffSelK];
                        for (uint32_t h = 0; h < fN1; ++h) {
                            Mul(prod, keyB, qRow[h*fDq], fDq, visible,
                                BinaryRepeatParams(1,1,1, fDq/8, fDq/8, 0));
                            WholeReduceSum(hSc[h*s2p], prod, fDq, visible, 1, 1, fDq/8);
                            Mul(prod, keyRB, qrRow[h*fDr], fDr, visible,
                                BinaryRepeatParams(1,1,1, fDr/8, fDr/8, 0));
                            WholeReduceSum(work, prod, fDr, visible, 1, 1, fDr/8);
                            Add(hSc[h*s2p], hSc[h*s2p], work, visible);
                        }
                        // a single shift for all heads keeps the exponentials in range
                        WholeReduceMax(work, hSc, visible, fN1, 1, 1, s2p/8);
                        WholeReduceMax(work[64], work, fN1, 1, 1, 1, 1);
                        float gmx = work[64].GetValue(0);
                        Adds(hSc, hSc, -gmx, visible, (uint8_t)fN1,
                             UnaryRepeatParams(1, 1, s2p/8, s2p/8));
                        Exp(hSc, hSc, visible, (uint8_t)fN1, UnaryRepeatParams(1, 1, s2p/8, s2p/8));
                        WholeReduceSum(work[128], hSc, visible, fN1, 1, 1, s2p/8);
                        Ln(work[128], work[128], fN1);
                        Adds(cRow, work[128], gmx, fN1);
                    } else {
                    for (uint32_t h = 0; h < fN1; ++h) {
                        Mul(prod, keyB, qRow[h*fDq], fDq, visible,
                            BinaryRepeatParams(1,1,1, fDq/8, fDq/8, 0));
                        WholeReduceSum(work, prod, fDq, visible, 1, 1, fDq/8);
                        Mul(prod, keyRB, qrRow[h*fDr], fDr, visible,
                            BinaryRepeatParams(1,1,1, fDr/8, fDr/8, 0));
                        WholeReduceSum(work[64], prod, fDr, visible, 1, 1, fDr/8);
                        Add(work, work, work[64], visible);
                        WholeReduceMax(work[128], work, visible, 1, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
                        float mx = work[128].GetValue(0);
                        Adds(work, work, -mx, visible);
                        Exp(work, work, visible);
                        WholeReduceSum(work[64], work, visible, 1, 1, 1, 1);
                        cRow.SetValue(h, mx + SliLog(work[64].GetValue(0)));
                    }
                    }
                }
                for (uint32_t j = 0; j < m; ++j) {
                    uint64_t kb = ((uint64_t)b*fS2 + sel[j]);
                    Mul(prod, qRow, base[fOffKey][kb*fDq], fDq, fN1, BinaryRepeatParams(1,1,1, fDq/8, fDq/8, 0));
                    WholeReduceSum(sco[j*fN1P], prod, fDq, fN1, 1, 1, fDq/8);
                    Mul(prod, qrRow, base[fOffKeyR][kb*fDr], fDr, fN1, BinaryRepeatParams(1,1,1, fDr/8, fDr/8, 0));
                    WholeReduceSum(work, prod, fDr, fN1, 1, 1, fDr/8);
                    Add(sco[j*fN1P], sco[j*fN1P], work, fN1);
                }
                Sub(sco, sco, cRow, fN1, m, BinaryRepeatParams(1, 1, 1, fN1P/8, fN1P/8, 0));
                Exp(sco, sco, fN1, m, UnaryRepeatParams(1, 1, fN1P/8, fN1P/8));
                WholeReduceSum(mass, sco, fN1, m, 1, 1, fN1P/8);
                // ---- indexer similarities ----
                Duplicate(sim, 0.0f, fKP*fNIP);
                for (uint32_t j = 0; j < m; ++j) {
                    uint64_t kb = ((uint64_t)b*fS2 + sel[j]);
                    Mul(prod, qiRow, base[fOffKi][kb*fDi], fDi, fNI, BinaryRepeatParams(1,1,1, fDi/8, fDi/8, 0));
                    WholeReduceSum(sim[j*fNIP], prod, fDi, fNI, 1, 1, fDi/8);
                    Relu(sim[j*fNIP], sim[j*fNIP], fNI);
                }
                Mul(prod, sim, wRow, fNI, m, BinaryRepeatParams(1, 1, 1, fNIP/8, fNIP/8, 0));
                WholeReduceSum(logit, prod, fNI, m, 1, 1, fNIP/8);
                // ---- softmax over the selected keys and KL ----
                WholeReduceSum(work, mass, m, 1, 1, 1, 1);
                float invM = 1.0f / work.GetValue(0);
                WholeReduceMax(work[128], logit, m, 1, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
                float logitMax = work[128].GetValue(0);
                Adds(work[192], logit, -logitMax, m);
                Exp(work[192], work[192], m);
                WholeReduceSum(work[64], work[192], m, 1, 1, 1, 1);
                float logL = logitMax + SliLog(work[64].GetValue(0));
                Muls(work[128], mass, invM, m);
                Maxs(work[192], work[128], 1e-30f, m);
                Ln(work[192], work[192], m);
                Sub(work[192], work[192], logit, m);
                Adds(work[192], work[192], logL, m);
                Mul(work[192], work[192], work[128], m);
                WholeReduceSum(work[64], work[192], m, 1, 1, 1, 1);
                Add(lossAcc, lossAcc, work[64], 1);
                Adds(logit, logit, -logL, m);
                Exp(logit, logit, m);
                Sub(di, logit, work[128], m);
                // per-key constants must vary along the repeat axis, so build a
                // row-constant matrix (row j = di[j]) before the reductions
                for (uint32_t p = 0; p < fNIP / 8; ++p)
                    Brcb(diMat[p*8], di, (uint8_t)((m + 7) / 8), BrcbRepeatParams(fNIP / 8, fNIP));
                // ---- dWeights ----
                // sum over keys (the row axis), so pad the key count to a power of two
                uint32_t mp = 1;
                while (mp < m) mp <<= 1;
                if (mp > m) Duplicate(prod, 0.0f, mp*fNIP);
                Mul(prod, sim, diMat, m*fNIP);
                LocalTensor<float> dwSrc = SumRows(prod, prod2, mp, fNIP);
                if (fNIP == fNI) {
                    Adds(dwRow, dwSrc, 0.0f, fNI);
                } else {
                    for (uint32_t h = 0; h < fNI; ++h) dwRow.SetValue(h, dwSrc.GetValue(h));
                }
                // ---- dQueryIndex / dKeyIndex ----
                Muls(pos, sim, 1.0e30f, m*fNIP);
                Mins(pos, pos, 1.0f, m*fNIP);
                Mul(vMat, pos, diMat, m*fNIP);
                Mul(uMat, vMat, wRow, fNI, m, BinaryRepeatParams(1, 1, 1, fNIP/8, fNIP/8, 0));
                Duplicate(dqAcc, 0.0f, fNIP*fDi);
                for (uint32_t j = 0; j < m; ++j) {
                    LocalTensor<float> uRowJ = uMat[j*fNIP];
                    for (uint32_t p = 0; p < fDi/8; ++p) {
                        Brcb(ubScratch[p*8], uRowJ, (uint8_t)(fNIP/8),
                             BrcbRepeatParams(fDi/8, fDi));
                    }
                    Mul(prod2, base[fOffKi][((uint64_t)b*fS2 + sel[j])*fDi], ubScratch, fDi, fNIP,
                        BinaryRepeatParams(1, 1, 1, fDi/8, 0, fDi/8));
                    Add(dqAcc, dqAcc, prod2, fNIP*fDi);
                    // dKeyIndex row for this key: sum over heads of u[j][h] * qi[h][:]
                    Mul(prod, qiRow, ubScratch, fDi, fNI, BinaryRepeatParams(1,1,1, fDi/8, fDi/8, fDi/8));
                    uint32_t left = fNI;
                    LocalTensor<float> cur = prod;
                    LocalTensor<float> nxt = prod2;
                    bool atProd = true;
                    while (left > 1) {
                        uint32_t half = left / 2;
                        Add(nxt, cur, cur[fDi], fDi, (uint8_t)half,
                            BinaryRepeatParams(1, 1, 1, fDi/8, 2*fDi/8, 2*fDi/8));
                        cur = nxt;
                        nxt = atProd ? prod2 : prod;
                        atProd = !atProd;
                        left = half;
                    }
                    LocalTensor<float> dkRow = base[fOffDk][((uint64_t)b*fS2 + sel[j])*fDi];
                    Add(dkRow, dkRow, cur, fDi);
                }
                Adds(dqRow, dqAcc, 0.0f, fNI*fDi);
            }
        }
        // ---- publish this core's share ----
        if (mine) StoreRowGradients(first, mine);
        WholeReduceSum(work[64], lossAcc, 1, 1, 1, 1, 1);
        float myLoss = totalLoss + work[64].GetValue(0);
        if (cores > 1) {
            // gradient accumulator and loss go through the per-core workspace slot
            DataCopyExtParams pw{1, static_cast<uint32_t>(keys*fDi*sizeof(float)), 0, 0, 0};
            SyncVToMte3();
            DataCopyPad(scratch, base[fOffDk], pw);
            PipeBarrier<PIPE_ALL>();
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(scratch);
            losses.SetValue(GetBlockIdx()*16, myLoss);
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(losses);
            SyncAll();
            if (GetBlockIdx() == 0) {
                // other cores' partials must come from GM, not from this core's cache
                DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(wsAll);
                PipeBarrier<PIPE_ALL>();
                uint32_t span = keys*fDi;
                uint32_t slot = t.batchScratch ? t.batchScratch : span;
                LocalTensor<float> acc = MergePartials(base, cores, span, slot, 0u);
                FastStore(dk, 0, acc, span);
                // the loss shares come back along with the partials
                LocalTensor<float> lossBuf = base[fOffAux];
                DataCopyExtParams pl{1, static_cast<uint32_t>(cores*16*sizeof(float)), 0, 0, 0};
                DataCopyPad(lossBuf, losses, pl, DataCopyPadExtParams<float>{false,0,0,0});
                SyncMte2ToV();
                SyncVToS();
                float total = 0;
                for (uint32_t c = 0; c < cores; ++c) total += lossBuf[c*16].GetValue(0);
                loss.SetValue(0, total);
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(loss);
            }
        } else {
            FastStore(dk, 0, base[fOffDk], keys*fDi);
            loss.SetValue(0, myLoss);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(loss);
        }
    }
    // ---- tiny path: one row per core, one contraction per selected key ----
    // Every dot is a broadcast multiply plus a single repeat reduce, and the
    // per-key constants shared by the gradient terms come out of one block
    // spread copy instead of one broadcast per key and head.
    __aicore__ inline void ProcessTiny() {
        LocalTensor<float> base = a;
        LocalTensor<int32_t> idxT = base[fOffIdx].ReinterpretCast<int32_t>();
        uint32_t keys = static_cast<uint32_t>((uint64_t)t.batch*fS2);
        uint32_t cores = t.cores ? t.cores : 1u;
        uint32_t per = (fRows + cores - 1) / cores;
        uint32_t first = GetBlockIdx() * per;
        uint32_t last = first + per;
        if (last > fRows) last = fRows;
        if (first > fRows) first = fRows;
        uint32_t mine = last - first;
        bool narrowAligned = (fNIP == fNI) && (fN1P == fN1);
        if (mine == 0) {
            // A core without a row has nothing to contribute and the row owner
            // folds only the producers, so it leaves without touching the workspace.
            return;
        }
        // A core's rows never straddle a batch here, so only that batch's key
        // rows have to be resident.
        uint32_t kb0 = t.queryRows ? first / t.queryRows : 0u;
        uint32_t kb1 = (last > first ? last - 1 : first) / (t.queryRows ? t.queryRows : 1u);
        bool oneBatch = (kb0 == kb1);
        uint32_t kbase = oneBatch ? kb0 : 0u;
        uint32_t kcount = oneBatch ? fS2 : keys;
        // every transfer is issued before a single pipeline wait
        FastDma(base[fOffQ], q, (uint64_t)first*fN1*fDq, mine*fN1*fDq, fStgQ);
        FastDma(base[fOffQR], qr, (uint64_t)first*fN1*fDr, mine*fN1*fDr, fStgQR);
        FastDma(base[fOffQI], qi, (uint64_t)first*fNI*fDi, mine*fNI*fDi, fStgQI);
        FastDma(base[fOffKey], k, (uint64_t)kbase*fS2*fDq, kcount*fDq, fStgKey);
        FastDma(base[fOffKeyR], kr, (uint64_t)kbase*fS2*fDr, kcount*fDr, fStgKeyR);
        FastDma(base[fOffKi], ki, (uint64_t)kbase*fS2*fDi, kcount*fDi, fStgKi);
        if (mine*fK) FastPad(idxT, indices, (uint64_t)first*fK, mine*fK);
        LocalTensor<int32_t> qlenT = base[fOffAux][1024].ReinterpretCast<int32_t>();
        LocalTensor<int32_t> klenT = base[fOffAux][1536].ReinterpretCast<int32_t>();
        if (cores > 1) {
            // the launch epoch rides along with the input transfers, so the handshake
            // at the end of the row work never waits on a fresh transfer
            LocalTensor<uint32_t> epochIn = base[fOffAux][kEpochLaneOffset].ReinterpretCast<uint32_t>();
            FastPad(epochIn, epochWord, 0, 1);
            if constexpr (AscendC::IsSameType<T,float>::value) {
                // the owner's clear is published just as early, so its word is worth
                // fetching with the inputs: the fold usually finds it already landed
                LocalTensor<uint32_t> clearIn = base[fOffAux][kStampLaneOffset].ReinterpretCast<uint32_t>();
                FastPad(clearIn, clearWord, 0, 1);
            }
        }
        if (t.queryLengthType == 1) FastPad(qlenT, ql32, 0, t.batch);
        else if (t.queryLengthType == 2) FastPad(qlenT.ReinterpretCast<int64_t>(), ql64, 0, t.batch);
        if (t.keyLengthType == 1) FastPad(klenT, kl32, 0, t.batch);
        else if (t.keyLengthType == 2) FastPad(klenT.ReinterpretCast<int64_t>(), kl64, 0, t.batch);
        if (narrowAligned) {
            if constexpr (AscendC::IsSameType<W,float>::value) {
                FastPad(base[fOffW], w, (uint64_t)first*fNI, mine*fNI);
            } else {
                FastPad(a[fOffWStage].ReinterpretCast<W>(), w, (uint64_t)first*fNI, mine*fNI);
            }
            if (!t.fastRecompute) {
                FastPad(base[fOffMax], maxima, (uint64_t)first*fN1, mine*fN1);
                FastPad(base[fOffSum], sums, (uint64_t)first*fN1, mine*fN1);
            }
        } else {
            for (uint32_t r = 0; r < mine; ++r) {
                uint32_t tr = first + r;
                if constexpr (AscendC::IsSameType<W,float>::value) {
                    FastPad(base[fOffW][r*fNIP], w, (uint64_t)tr*fNI, fNI);
                } else {
                    FastPad(a[fOffWStage].ReinterpretCast<W>()[r*fWStageStride], w, (uint64_t)tr*fNI, fNI);
                }
                if (!t.fastRecompute) {
                    FastPad(base[fOffMax][r*fN1P], maxima, (uint64_t)tr*fN1, fN1);
                    FastPad(base[fOffSum][r*fN1P], sums, (uint64_t)tr*fN1, fN1);
                }
            }
        }
        SyncMte2ToV();
        FastCast<T>(base[fOffQ], fStgQ, mine*fN1*fDq);
        FastCast<QR>(base[fOffQR], fStgQR, mine*fN1*fDr);
        FastCast<T>(base[fOffQI], fStgQI, mine*fNI*fDi);
        if constexpr (!AscendC::IsSameType<W,float>::value) {
            if (narrowAligned) {
                Cast(base[fOffW], a[fOffWStage].ReinterpretCast<W>(), RoundMode::CAST_NONE,
                     static_cast<int32_t>(mine*fNI));
            } else {
                for (uint32_t r = 0; r < mine; ++r)
                    Cast(base[fOffW][r*fNIP], a[fOffWStage].ReinterpretCast<W>()[r*fWStageStride],
                         RoundMode::CAST_NONE, static_cast<int32_t>(fNI));
            }
        }
        FastCast<T>(base[fOffKey], fStgKey, kcount*fDq);
        FastCast<KR>(base[fOffKeyR], fStgKeyR, kcount*fDr);
        FastCast<T>(base[fOffKi], fStgKi, kcount*fDi);
        // the attention scale is applied to the scores, so the row vectors are
        // transferred and kept unscaled
        // The three accumulators sit in one contiguous span, so a single burst
        // clears all of them; a core without rows only publishes its key share.
        if (mine) {
            Duplicate(base[fOffDk], 0.0f, (uint64_t)fOffDw + fRows*fNIP - fOffDk);
        } else {
            Duplicate(base[fOffDk], 0.0f, (uint64_t)t.batch*fS2*fDi);
        }
        if (!t.fastRecompute) {
            Ln(base[fOffC], base[fOffSum], fRows*fN1P);
            Add(base[fOffC], base[fOffC], base[fOffMax], fRows*fN1P);
        }
        if (cores > 1 && AscendC::IsSameType<T,float>::value) {
            // The FP32 gradients fold straight into the outputs, so the row owner
            // clears them and publishes that clear before anyone adds to them.
            PipeBarrier<PIPE_ALL>();
            launchEpoch = base[fOffAux][kEpochLaneOffset].ReinterpretCast<uint32_t>().GetValue(0);
            if (GetBlockIdx() == 0) {
                FastStore(dk, 0, base[fOffDk], static_cast<uint32_t>((uint64_t)t.batch*fS2*fDi));
                loss.SetValue(0, 0.0f);
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(loss);
                LocalTensor<uint32_t> clearOut = base[fOffAux][kStampLaneOffset].ReinterpretCast<uint32_t>();
                clearOut.SetValue(0, SliStamp(launchEpoch, kClearStamp));
                SyncSToMte3();
                DataCopyExtParams pc{1, sizeof(uint32_t), 0, 0, 0};
                DataCopyPad(clearWord, clearOut, pc);
            }
        }
        LocalTensor<float> lossVec = base[fOffAux][512];
        LocalTensor<float> dIvec = base[fOffAux][128];
        LocalTensor<float> massV = base[fOffAux][256];
        LocalTensor<float> logitV = base[fOffAux][256 + fKP];
        LocalTensor<float> dsspr = base[fOffDsSpr];
        LocalTensor<float> dispr = base[fOffDiSpr];
        LocalTensor<float> prod = base[fOffProd];
        LocalTensor<float> prod2 = base[fOffProd][fN1P*fDq + 64];
        LocalTensor<float> sco = base[fOffSco];
        LocalTensor<float> sim = base[fOffSim];
        LocalTensor<float> ind = base[fOffPos];
        LocalTensor<float> ds = base[fOffDiMat];
        LocalTensor<float> dqRow = base[fOffDq];
        uint32_t sel[8];
        float totalLoss = 0.0f;
        // The supplied statistics may come from a lower-precision forward pass and
        // then no longer reproduce the exact visible normalizer, so the FP32 path
        // rebuilds every head's normalizer over the visible window instead.
        bool useProvided = !t.fastRecompute;
        for (uint32_t b = 0; b < t.batch; ++b) {
            uint32_t qlen = t.empty ? 0 : CachedLength(true, b, qlenT);
            uint32_t klen = CachedLength(false, b, klenT);
            for (uint32_t row = 0; row < t.queryRows; ++row) {
                uint64_t token = (uint64_t)b*t.queryRows + row;
                if (token < first || token >= last) continue;
                if (row >= qlen) continue;
                uint32_t local = (uint32_t)(token - first);
                uint32_t kbase = oneBatch ? 0u : static_cast<uint32_t>(b)*fS2;
                int32_t visible = static_cast<int32_t>(row) + static_cast<int32_t>(klen)
                                  - static_cast<int32_t>(qlen) + 1;
                uint32_t m = 0;
                for (uint32_t j = 0; j < fK; ++j) {
                    int32_t key = idxT.GetValue((uint64_t)local*fK + j);
                    if (key >= 0 && key < static_cast<int32_t>(klen) && key < visible) sel[m++] = (uint32_t)key;
                }
                if (m == 0) continue;
                LocalTensor<float> qRow = base[fOffQ][(uint64_t)local*fN1*fDq];
                LocalTensor<float> qrRow = base[fOffQR][(uint64_t)local*fN1*fDr];
                LocalTensor<float> qiRow = base[fOffQI][(uint64_t)local*fNI*fDi];
                LocalTensor<float> dwRow = base[fOffDw][(uint64_t)local*fNIP];
                LocalTensor<float> wRow = base[fOffW][(uint64_t)local*fNIP];
                LocalTensor<float> cRow = base[fOffC][(uint64_t)local*fN1P];
                LocalTensor<float> dqOut = dqRow[(uint64_t)local*fNI*fDi];
                if (!useProvided && visible > 0) {
                    // Stage every head's window scores in one matrix so the head
                    // reductions become a single strided pass and the shift can be
                    // shared by all heads.
                    LocalTensor<float> keyB = base[fOffKey][(uint64_t)kbase*fS2*fDq];
                    LocalTensor<float> keyRB = base[fOffKeyR][(uint64_t)kbase*fS2*fDr];
                    LocalTensor<float> hSc = base[fOffHeadScr];
                    uint32_t vis = static_cast<uint32_t>(visible);
                    uint32_t s2p = (fS2 + 7) / 8 * 8;
                    uint32_t hstride = s2p / 8;
                    for (uint32_t h = 0; h < fN1; ++h) {
                        // the rope block folds straight into the first lanes of the
                        // head rows, so one primitive and one reduce score a head
                        Mul(prod, keyB, qRow[h*fDq], fDq, vis,
                            BinaryRepeatParams(1, 1, 1, fDq/8, fDq/8, 0));
                        MulAddDst(prod, keyRB, qrRow[h*fDr], fDr, vis,
                                  BinaryRepeatParams(1, 1, 1, fDq/8, fDr/8, 0));
                        WholeReduceSum(hSc[h*s2p], prod, fDq, vis, 1, 1, fDq/8);
                    }
                    Muls(hSc, hSc, t.scale, vis, (uint8_t)fN1, UnaryRepeatParams(1, 1, hstride, hstride));
                    WholeReduceMax(lossVec[128], hSc, vis, (uint8_t)fN1, 1, 1, hstride,
                                   ReduceOrder::ORDER_ONLY_VALUE);
                    WholeReduceMax(lossVec[192], lossVec[128], fN1, 1, 1, 1, 1,
                                   ReduceOrder::ORDER_ONLY_VALUE);
                    float gmx = lossVec[192].GetValue(0);
                    Adds(hSc, hSc, -gmx, vis, (uint8_t)fN1, UnaryRepeatParams(1, 1, hstride, hstride));
                    Exp(hSc, hSc, vis, (uint8_t)fN1, UnaryRepeatParams(1, 1, hstride, hstride));
                    WholeReduceSum(lossVec[128], hSc, vis, (uint8_t)fN1, 1, 1, hstride);
                    Ln(lossVec[128], lossVec[128], fN1);
                    Adds(cRow, lossVec[128], gmx, fN1);
                }
                // ---- main scores and indexer similarities at the selected keys ----
                for (uint32_t j = 0; j < m; ++j) {
                    uint64_t kb = (uint64_t)kbase + sel[j];
                    Mul(prod, qRow, base[fOffKey][kb*fDq], fDq, (uint8_t)fN1,
                        BinaryRepeatParams(1, 1, 1, fDq/8, fDq/8, 0));
                    MulAddDst(prod, qrRow, base[fOffKeyR][kb*fDr], fDr, (uint8_t)fN1,
                              BinaryRepeatParams(1, 1, 1, fDq/8, fDr/8, 0));
                    WholeReduceSum(sco[j*fN1P], prod, fDq, (uint8_t)fN1, 1, 1, fDq/8);
                    Mul(prod, qiRow, base[fOffKi][kb*fDi], fDi, (uint8_t)fNI,
                        BinaryRepeatParams(1, 1, 1, fDi/8, fDi/8, 0));
                    WholeReduceSum(sim[j*fNIP], prod, fDi, (uint8_t)fNI, 1, 1, fDi/8);
                }
                Muls(sco, sco, t.scale, m*fN1P);
                uint32_t mMask = m*fNIP;
                Maxs(sim, sim, 0.0f, mMask);
                Muls(ind, sim, 1e30f, mMask);
                Mins(ind, ind, 1.0f, mMask);
                // ---- selected softmax and KL ----
                Mul(prod2, sim, wRow, fNI, (uint8_t)m, BinaryRepeatParams(1, 1, 1, fNIP/8, fNIP/8, 0));
                WholeReduceSum(logitV, prod2, fNI, (uint8_t)m, 1, 1, fNIP/8);
                Sub(prod2, sco, cRow, fN1, (uint8_t)m, BinaryRepeatParams(1, 1, 1, fN1P/8, fN1P/8, 0));
                Exp(prod2, prod2, fN1, (uint8_t)m, UnaryRepeatParams(1, 1, fN1P/8, fN1P/8));
                WholeReduceSum(massV, prod2, fN1, (uint8_t)m, 1, 1, fN1P/8);
                SyncVToS();
                float mass[8], logit[8], di[8];
                float msum = 0.0f, lmax = -3.4e38f;
                for (uint32_t j = 0; j < m; ++j) {
                    mass[j] = massV.GetValue(j);
                    msum += mass[j];
                    logit[j] = logitV.GetValue(j);
                    if (logit[j] > lmax) lmax = logit[j];
                }
                if (!(msum > 0.0f)) msum = 1.0f;
                float lsum = 0.0f;
                for (uint32_t j = 0; j < m; ++j) lsum += SliExp(logit[j] - lmax);
                float logl = lmax + SliLog(lsum);
                for (uint32_t j = 0; j < m; ++j) {
                    float p = mass[j]/msum;
                    float q = SliExp(logit[j] - logl);
                    di[j] = q - p;
                    if (p > 0.0f) totalLoss += p*(SliLog(p) - (logit[j] - logl));
                }
                for (uint32_t j = 0; j < m; ++j) dIvec.SetValue(j, di[j]);
                SyncSToV();
                // ---- gradients ----
                // dWeights uses the rectified similarities themselves, the two
                // scatter terms only their sign, so the shared product carries the
                // indicator and the weights while dWeights keeps the plain product.
                Brcb(dispr, dIvec, 1, BrcbRepeatParams(1, 8));
                Mul(ds, ind, dispr, fNI, (uint8_t)m,
                    BinaryRepeatParams(1, 1, 0, fNIP/8, fNIP/8, 1));
                Mul(ds, ds, wRow, fNI, (uint8_t)m, BinaryRepeatParams(1, 1, 1, fNIP/8, fNIP/8, 0));
                // dWeights contracts over the keys and dKeyIndex over the heads, so
                // each keeps one live accumulator and folds every term into it.
                MulAddDst(dwRow, sim, dispr, fNI, (uint8_t)m,
                          BinaryRepeatParams(1, 1, 0, 0, fNIP/8, 1));
                Brcb(dsspr, ds, (uint8_t)((m*fNIP)/8), BrcbRepeatParams(1, 8));
                // The key accumulator is zeroed once for the whole core, so every
                // scatter term folds straight into its own key row.
                for (uint32_t j = 0; j < m; ++j) {
                    uint64_t kb = (uint64_t)kbase + sel[j];
                    MulAddDst(dqOut, base[fOffKi][kb*fDi], dsspr[(uint64_t)j*fNIP*8], fDi, (uint8_t)fNI,
                              BinaryRepeatParams(1, 1, 0, fDi/8, 0, 1));
                    MulAddDst(base[fOffDk][((uint64_t)b*fS2 + sel[j])*fDi], qiRow,
                              dsspr[(uint64_t)j*fNIP*8], fDi, (uint8_t)fNI,
                              BinaryRepeatParams(1, 1, 0, 0, fDi/8, 1));
                }
            }
        }
        if (mine) StoreRowGradients(first, mine);
        PublishTiny(totalLoss);
    }
    // Publish this core's share of the merged gradient and of the loss.  The
    // vector-only launch carries no hardware barrier, so the producers rendezvous
    // through the workspace: each one leaves its partial, its loss share and a
    // ready stamp derived from the launch epoch in a single transfer, and the row
    // owner folds the partials once every stamp of this launch has landed.
    __aicore__ inline void PublishTiny(float myLoss) {
        LocalTensor<float> base = a;
        LocalTensor<uint32_t> auxWords = base[fOffAux][kEpochLaneOffset].ReinterpretCast<uint32_t>();
        uint32_t keys = static_cast<uint32_t>((uint64_t)t.batch*fS2);
        uint32_t cores = t.cores ? t.cores : 1u;
        uint32_t span = keys*fDi;
        if (cores > 1 && AscendC::IsSameType<T,float>::value) {
            // the row owner has cleared the accumulators; fold this core's share in.
            // Its own clear already completed before it published the stamp, so the
            // owner only has to confirm the word it prefetched with the inputs.
            LocalTensor<uint32_t> clearIn = base[fOffAux][kStampLaneOffset].ReinterpretCast<uint32_t>();
            uint32_t want = SliStamp(launchEpoch, kClearStamp);
            PipeBarrier<PIPE_ALL>();
            uint32_t settled = GetBlockIdx() == 0 || clearIn.GetValue(0) == want;
            while (!settled) {
                SyncVToMte2();
                DataCopyExtParams pr{1, sizeof(uint32_t), 0, 0, 0};
                DataCopyPad(clearIn, clearWord, pr, DataCopyPadExtParams<uint32_t>{false,0,0,0});
                PipeBarrier<PIPE_ALL>();
                settled = clearIn.GetValue(0) == want;
                PipeBarrier<PIPE_ALL>();
            }
            base[fOffAux][kLossLane].SetValue(0, myLoss);
            PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<float>();
            FastStore(dk, 0, base[fOffDk], span);
            DataCopyExtParams pl{1, sizeof(float), 0, 0, 0};
            DataCopyPad(loss[0], base[fOffAux][kLossLane], pl);
            SetAtomicNone();
            if (GetBlockIdx() == 0) {
                // the next launch reads this before any core relies on the clear
                LocalTensor<uint32_t> epochOut = base[fOffAux][kEpochLaneOffset].ReinterpretCast<uint32_t>();
                epochOut.SetValue(0, launchEpoch + 1u);
                SyncSToMte3();
                DataCopyExtParams pe{1, sizeof(uint32_t), 0, 0, 0};
                DataCopyPad(epochWord, epochOut, pe);
            }
        } else if (cores > 1) {
            uint32_t per = (fRows + cores - 1) / cores;
            uint32_t active = per ? (fRows + per - 1) / per : 0u;
            if (active > cores) active = cores;
            uint32_t slot = t.batchScratch ? t.batchScratch : span;
            // the epoch transfer was issued with the preamble, so it is long settled
            PipeBarrier<PIPE_ALL>();
            uint32_t epoch = auxWords.GetValue(kEpochLane);
            // The partial goes out on its own, and the loss lane with the ready stamp
            // follows in a second transfer.  A reader that sees the stamp can then
            // trust the partial: the segments of a single burst are not guaranteed to
            // land in order, but queued transfers are issued in order.
            DataCopyExtParams pw{1, static_cast<uint32_t>(span*sizeof(float)), 0, 0, 0};
            DataCopyPad(scratch, base[fOffDk], pw);
            Duplicate(base[fOffDk][span], myLoss, 1);
            base[fOffDk].ReinterpretCast<uint32_t>().SetValue(span + 1, SliStamp(epoch, GetBlockIdx()));
            SyncVToMte3();
            SyncSToMte3();
            // a full 32B block, so the closing transfer keeps the copy granularity
            DataCopyExtParams pz{1, 8*sizeof(float), 0, 0, 0};
            DataCopyPad(scratch[span], base[fOffDk][span], pz);
            if (GetBlockIdx() == 0) {
                // One copy brings in every core's partial together with the loss lane
                // and the ready stamp that closes it, so the fold waits on a single
                // round trip: the stamp travels last in the producer's burst, and a
                // partial whose stamp is visible is therefore complete.
                LocalTensor<float> buf = base[fOffRed][span + 8];
                LocalTensor<uint32_t> stampIn = buf.ReinterpretCast<uint32_t>();
                uint32_t expect[8];
                for (uint32_t c = 0; c < active; ++c) expect[c] = SliStamp(epoch, c);
                uint32_t settled = 0;
                while (settled < active) {
                    SyncVToMte2();
                    DataCopyExtParams pr{1, static_cast<uint32_t>(active*slot*sizeof(float)), 0, 0, 0};
                    DataCopyPad(buf, wsAll, pr, DataCopyPadExtParams<float>{false,0,0,0});
                    SyncMte2ToS();
                    settled = 0;
                    for (uint32_t c = 0; c < active; ++c)
                        if (stampIn.GetValue((uint64_t)c*slot + span + 1) == expect[c]) ++settled;
                }
                uint32_t left = active;
                while (left > 1) {
                    uint32_t half = left / 2;
                    Add(buf, buf, buf[(uint64_t)half*slot], half*slot);
                    if (left & 1u) {
                        Adds(buf[(uint64_t)half*slot], buf[(uint64_t)(left - 1)*slot], 0.0f, slot);
                        left = half + 1;
                    } else {
                        left = half;
                    }
                }
                FastStore(dk, 0, buf, span);
                // the merged loss lane leaves through the copy path as well
                SyncVToMte3();
                DataCopyExtParams pl{1, sizeof(float), 0, 0, 0};
                DataCopyPad(loss[0], buf[span], pl);
                // The next launch reads this before any core stamps again.  It has to
                // wait until the fold is done: a core that is still being dispatched
                // has not fetched this launch's epoch yet, and a value that moves
                // under it would make its stamps miss forever.
                auxWords.SetValue(kEpochLane, epoch + 1u);
                SyncSToMte3();
                DataCopyExtParams pe{1, sizeof(uint32_t), 0, 0, 0};
                DataCopyPad(epochWord, auxWords, pe);
            }
        } else {
            FastStore(dk, 0, base[fOffDk], keys*fDi);
            loss.SetValue(0, myLoss);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(loss);
        }
    }
    __aicore__ inline void Process() {
        if (t.fast) {
            if (t.tiny) { ProcessTiny(); return; }
            ProcessFast();
            return;
        }
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
    // the row-split fast path synchronizes cores with the hardware barrier, which
    // needs the MIX AIV:AIC 1:0 launch type (it still starts vector cores only)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SparseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseLightningIndexerGradKlLossTilingData,t,tiling);
    SparseGrad<DT_QUERY,DT_WEIGHT,DT_QROPE,DT_KROPE> op;
    TPipe pipe;
    op.Init(query,key,queryIndex,keyIndex,weights,sparseIndices,softmaxMax,softmaxSum,
        queryRope,keyRope,actualQ,actualK,dQuery,dKey,dWeight,loss,GetUserWorkspace(workspace),&pipe,t);
    op.Process();
}
