from sdks.novavision.src.helper.package import PackageHelper
from capsules.CcvGmm.src.models.PackageModel import (
    PackageModel,
    OutputData,
    ConfigExecutor,
    PackageConfigs,
    GmmExecutor,
    GmmOutputs,
    GmmResponse,
)


def build_response_gmm(context):

    outputData = OutputData(value=context.outputData)

    Outputs = GmmOutputs(outputData=outputData)

    packageResponse = GmmResponse(outputs=Outputs)
    packageExecutor = GmmExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)

    return packageModel
