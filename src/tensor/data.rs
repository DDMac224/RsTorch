use crate::tensor::{Device, Element, cpu::CpuData, metadata::TensorMetadata};

#[derive(Debug, Clone)]
pub enum TensorData<T> {
    CpuData(CpuData<T>),
    CudaData,
}

impl<T> TensorData<T>
where
    T: Element,
{
    pub fn new(data: Vec<T>, device: Device) -> Self {
        match device {
            Device::CPU => {
                return TensorData::CpuData(CpuData(data));
            }
            Device::Cuda => todo!(),
        }
    }

    pub fn item(&self, metadata: &TensorMetadata) -> T {
        match self {
            TensorData::CpuData(data) => data.item(metadata),
            TensorData::CudaData => todo!(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::CpuData(data) => data.len(),
            TensorData::CudaData => todo!(),
        }
    }

    pub fn contiguous(&self, metadata: &TensorMetadata) -> Self {
        match self {
            TensorData::CpuData(data) => Self::CpuData(data.contiguous(metadata)),
            TensorData::CudaData => todo!(),
        }
    }
}
