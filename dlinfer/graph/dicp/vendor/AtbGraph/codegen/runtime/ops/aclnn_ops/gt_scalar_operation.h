#pragma once

#include "acl_nn_operation.h"
#include "utils/scalar.h"

namespace dicp {

class AclNnGtScalarOperation : public AclNnOperation {
public:
    explicit AclNnGtScalarOperation(const std::string& name, const std::string& value, const std::string& dtype);
    ~AclNnGtScalarOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs, atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    DICPScalar other_;
    aclScalar* aclOther_ = nullptr;
    bool need_update_value_;
    std::string value_;
    int SetAclNnWorkspaceExecutor(uint64_t& workspaceSize) override;
    int CallAclExecute(uint8_t* workspace, uint64_t workspaceSize, aclOpExecutor* aclExecutor, aclrtStream stream) override;
};

}  // namespace dicp
