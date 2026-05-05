use std::sync::Arc;

use crate::tensor::{Element, Tensor, TensorInner, metadata::TensorMetadata};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn is_contiguous(&self) -> bool {
        let mut expt_strd: usize = 1;

        for (&strd, &shp) in self
            .metadata
            .stride
            .iter()
            .rev()
            .zip(self.metadata.shape.iter().rev())
        {
            if strd != expt_strd {
                return false;
            }
            expt_strd *= shp;
        }
        return true;
    }

    pub fn contiguous(&self) -> Self {
        let self_data_locked = self.data.read().unwrap();

        Tensor::from_parts(
            self_data_locked.contiguous(&self.metadata),
            TensorMetadata::new(self.metadata.shape.clone()),
            None,
            self.requires_grad(),
        )
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert!(
            shape.iter().product::<usize>() == self.shape().iter().product(),
            "Shape of: {:?} is not compatible with tensor of size: {:?}.",
            shape,
            self.shape().iter().product::<usize>()
        );

        let mut data = self.data.clone();
        if !self.is_contiguous() {
            let cont = self.contiguous();
            data = cont.data.clone();
        }

        Self(Arc::new(TensorInner {
            data: data,
            metadata: self.metadata.reshape(shape),
            grad_node: self.grad_node.clone(),
            requires_grad: self.requires_grad,
        }))
    }

    pub fn transpose(&self) -> Self {
        let mut data = self.data.clone();
        if !self.is_contiguous() {
            let cont = self.contiguous();
            data = cont.data.clone();
        }

        Tensor(Arc::new(TensorInner {
            data: data,
            metadata: self.metadata.transpose(),
            grad_node: self.grad_node.clone(),
            requires_grad: self.requires_grad,
        }))
    }
}
