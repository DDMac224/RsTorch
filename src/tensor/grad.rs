use std::sync::{Arc, OnceLock, RwLock};

use crate::{
    autograd::node::GradNode,
    tensor::{Device, Element, Tensor, TensorInner, data::TensorData, metadata::TensorMetadata},
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

    pub fn grad(&self) -> Option<Tensor<T>> {
        self.grad_node
            .get()
            .map(|node| node.grad.lock().unwrap().clone())
            .flatten()
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn backward(&self) {
        assert!(self.requires_grad, "Tensor does not require gradient.");

        let seed = Tensor::ones(self.shape(), Some(self.device()), None);

        if let Some(node) = self.grad_node.get() {
            node.backward(seed);
        } else {
            panic!("Tensor does not have grad_node")
        }
    }

    pub fn detach(&self) -> Tensor<T> {
        let grad_node = OnceLock::new();
        let _ = grad_node.set(GradNode::leaf());

        Self(Arc::new(TensorInner {
            data: Arc::clone(&self.data),
            metadata: self.metadata.clone(),
            grad_node,
            requires_grad: self.requires_grad(),
        }))
    }
}
