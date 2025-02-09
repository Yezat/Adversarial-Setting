from enum import Enum


class DataModelType(Enum):
    VanillaGaussian = 1
    SourceCapacity = 2
    MarginGaussian = 3
    KFeaturesModel = 4


class SigmaDeltaProcessType(Enum):
    UseContent = 0
    ComputeTeacherOrthogonal = 1
    ComputeTeacherDirection = 2
