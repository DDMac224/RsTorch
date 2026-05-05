use std::{
    cmp,
    sync::{Arc, OnceLock},
};

use itertools::Itertools;

use crate::tensor::{Element, Tensor, TensorInner};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn broadcast_to(&self, target: &Vec<usize>) -> Self {
        Self(Arc::new(TensorInner {
            data: self.data.clone(),
            metadata: self.metadata.broadcast_to(target),
            grad_node: OnceLock::new(),
            requires_grad: self.requires_grad,
        }))
    }

    pub fn broadcast_tensors(&self, other: &Self) -> (Self, Self) {
        let new_metadata = self.metadata.broadcast_tensors(&other.metadata);

        (
            Self(Arc::new(TensorInner {
                data: self.data.clone(),
                metadata: new_metadata.0,
                grad_node: OnceLock::new(),
                requires_grad: self.requires_grad,
            })),
            Self(Arc::new(TensorInner {
                data: self.data.clone(),
                metadata: new_metadata.1,
                grad_node: OnceLock::new(),
                requires_grad: self.requires_grad,
            })),
        )
    }
}
