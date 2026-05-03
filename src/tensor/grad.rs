use std::sync::{Arc, OnceLock};

use crate::tensor::{Device, Element, Tensor};

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

        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        Self {
            data: Arc::new(data),
            stride,
            shape: shape,
            device: device,
            offset: 0,
            grad_node: OnceLock::new(),
            requires_grad: true,
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn backward(&self) {
        assert!(self.requires_grad, "Tensor does not require gradient.");
        if let Some(node) = &self.grad_node.get() {
            node.backward();
        } else {
            panic!("Tensor does not have grad_node")
        }
    }
}
