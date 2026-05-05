use std::sync::{Arc, OnceLock, RwLock};

use crate::tensor::{
    Device, Element, Tensor, TensorInner, data::TensorData, metadata::TensorMetadata,
};

impl<T> Tensor<T>
where
    T: Element,
{
    pub(crate) fn new_from_op(data: Vec<T>, shape: Vec<usize>, device: Device) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Length of data does not match shape"
        );

        Tensor(Arc::new(TensorInner {
            data: Arc::new(RwLock::new(TensorData::new(data, device))),
            metadata: TensorMetadata::new(shape),
            grad_node: OnceLock::new(),
            requires_grad: true,
        }))
    }

    pub fn grad(&self) -> Tensor<T> {
        //self.grad_node.get().unwrap().grad.lock();
        todo!()
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn backward(&self) {
        assert!(self.requires_grad, "Tensor does not require gradient.");

        if let Some(node) = self.grad_node.get() {
            node.accumulate_grad(self);
        } else {
            panic!("Tensor does not have grad_node")
        }
    }
}
