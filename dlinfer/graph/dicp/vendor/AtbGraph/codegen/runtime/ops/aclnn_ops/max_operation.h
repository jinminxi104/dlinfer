#pragma once
#include "acl_nn_operation.h"

namespace dicp {
class AclNnMaxOperation : public AclNnOperation {
public:
    explicit AclNnMaxOperation(const std::string& name);
    ~AclNnMaxOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

inline atb::Operation* AclNnMaxOperationCreate(const nlohmann::json& paramJson) {
    std::string opName;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (paramJson.contains("name")) {
        opName = paramJson["name"].get<std::string>();
    }
    DICP_LOG(INFO) << "AclNnMaxOperation name: " << opName;
    atb::Operation* op = new AclNnMaxOperation(opName);
    return op;
}

}  // namespace dicp
