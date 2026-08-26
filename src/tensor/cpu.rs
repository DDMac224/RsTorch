use crate::tensor::data::TensorData;
use crate::tensor::{Element, metadata::TensorMetadata};

pub mod ops;

#[derive(Debug, Clone)]
pub struct CpuData<T>(pub(crate) Vec<T>);

impl<T> CpuData<T>
where
    T: Element,
{
    pub fn item(&self, metadata: &TensorMetadata) -> T {
        assert!(metadata.is_scalar(), "Tensor must be scalar");

        self.0[metadata.offset]
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn contiguous(&self, metadata: &TensorMetadata) -> Self {
        let size = metadata.size();
        let mut data: Vec<T> = Vec::new();

        for i in 0..size {
            data.push(
                self.0[metadata.offset
                    + metadata
                        .shape
                        .iter()
                        .rev()
                        .scan(i, |acc, e| {
                            let temp = *acc;
                            *acc /= *e;
                            Some(temp % e)
                        })
                        .zip(metadata.stride.iter().rev())
                        .fold(0, |acc, (idx, strd)| acc + (idx * strd))],
            );
        }

        CpuData(data)
    }
}

impl<T> TensorData<T>
where
    T: Element,
{
    pub fn expect_cpu(&self) -> &[T] {
        match self {
            TensorData::CpuData(CpuData(vec)) => vec,
            _ => panic!("Data is not on Cpu"),
        }
    }
}
