use crate::{
    autograd::{BackwardFn, node::GradNode, ops::op_impl},
    tensor::{Element, Tensor},
};

use std::{ops::Add, sync::Arc};

#[derive(Debug, Clone)]
pub struct AddBackward<T: Element> {
    lhs: Arc<Tensor<T>>,
    rhs: Arc<Tensor<T>>,
}

impl<T> BackwardFn<T> for AddBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>) {
        self.lhs
            .grad_node
            .get()
            .unwrap()
            .call_backwardfn(Arc::clone(&self.lhs));
        self.rhs
            .grad_node
            .get()
            .unwrap()
            .call_backwardfn(Arc::clone(&self.rhs));
    }

    fn zero_grad(&self) {
        self.lhs.grad_node.get().unwrap().zero_grad();
        self.rhs.grad_node.get().unwrap().zero_grad();
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    fn add_grad(&self, rhs: &Self) -> Self {
        let ret = self.elemwise_add(rhs);

        let backwrd_fn = GradNode::new(Box::new(AddBackward {
            lhs: Arc::new(self.clone()),
            rhs: Arc::new(rhs.clone()),
        }));

        if self.requires_grad() || rhs.requires_grad() {
            let _ = ret.grad_node.set(backwrd_fn);
        }

        ret
    }
}

op_impl!(Add, add; add_grad, AddBackward);

#[derive(Debug, Clone)]
pub struct MatMulBackward<T: Element> {
    lhs: Arc<Tensor<T>>,
    rhs: Arc<Tensor<T>>,
}

impl<T> BackwardFn<T> for MatMulBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>) {
        self.lhs
            .grad_node
            .get()
            .unwrap()
            .call_backwardfn(Arc::new(grad_output.forward_matmul(&self.rhs.transpose())));
        self.rhs
            .grad_node
            .get()
            .unwrap()
            .call_backwardfn(Arc::new(self.lhs.transpose().forward_matmul(&grad_output)));
    }

    fn zero_grad(&self) {
        self.lhs.grad_node.get().unwrap().zero_grad();
        self.rhs.grad_node.get().unwrap().zero_grad();
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    fn matmul_grad(&self, rhs: &Self) -> Self {
        let ret = self.forward_matmul(rhs);

        let backwrd_fn = GradNode::new(Box::new(MatMulBackward {
            lhs: Arc::new(self.clone()),
            rhs: Arc::new(rhs.clone()),
        }));

        if self.requires_grad() || rhs.requires_grad() {
            let _ = ret.grad_node.set(backwrd_fn);
        }

        ret
    }
}

op_impl!(matmul; matmul_grad, MatMulBackward);
