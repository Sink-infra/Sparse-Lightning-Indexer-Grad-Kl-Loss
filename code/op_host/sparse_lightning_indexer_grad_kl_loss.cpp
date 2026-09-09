// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "../op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_query = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_key = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_queryIndex = context->GetRequiredInputTensor(2);
        const gert::Tensor *tensor_keyIndex = context->GetRequiredInputTensor(3);
        const gert::Tensor *tensor_weights = context->GetRequiredInputTensor(4);
        const gert::Tensor *tensor_sparseIndices = context->GetRequiredInputTensor(5);
        const gert::Tensor *tensor_softmaxMax = context->GetRequiredInputTensor(6);
        const gert::Tensor *tensor_softmaxSum = context->GetRequiredInputTensor(7);
        const gert::Tensor *tensor_queryRope = context->GetRequiredInputTensor(8);
        const gert::Tensor *tensor_keyRope = context->GetRequiredInputTensor(9);
        const gert::Tensor *tensor_actual_seq_lengths_query = context->GetOptionalInputTensor(10);
        if(tensor_actual_seq_lengths_query) {
            // 可选输入数组存在
        }
        const gert::Tensor *tensor_actual_seq_lengths_key = context->GetOptionalInputTensor(11);
        if(tensor_actual_seq_lengths_key) {
            // 可选输入数组存在
        }
        ge::DataType dtype_query = tensor_query->GetDataType(); // 获取数据类型
        int dtype_size_query = ge::GetSizeByDataType(dtype_query); // 获取数据类型的字长
        uint32_t length_query = tensor_query->GetShapeSize(); // 获取元素个数
        uint32_t size_query = tensor_query->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_scale_value = attrs->GetFloat(0);
        const char *attr_layout = attrs->GetStr(1);
        const int64_t *attr_sparse_mode = attrs->GetInt(2);
        const bool *attr_deterministic = attrs->GetBool(3);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_QUERY = static_cast<uint32_t>(dtype_query);
        ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);
        // 示例: 计算tiling方案并填充tiling结构体
        SparseLightningIndexerGradKlLossTilingData *tiling = context->GetTilingData<SparseLightningIndexerGradKlLossTilingData>();
        tiling->length = length_query;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SparseLightningIndexerGradKlLoss : public OpDef {
    public:
        explicit SparseLightningIndexerGradKlLoss(const char *name) : OpDef(name) {
            this->Input("query")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("key")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("queryIndex")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("keyIndex")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("weights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("sparseIndices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("softmaxMax")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("softmaxSum")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("queryRope")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("keyRope")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("actual_seq_lengths_query")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("actual_seq_lengths_key")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("dQueryIndex")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("dKeyIndex")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("dWeights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("loss")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0442);
            this->Attr("layout").AttrType(OPTIONAL).String("BSND");
            this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
            this->Attr("deterministic").AttrType(OPTIONAL).Bool();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SparseLightningIndexerGradKlLoss);
}  // namespace ops
