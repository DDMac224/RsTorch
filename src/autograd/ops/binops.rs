use crate::{
    autograd::{BackwardFn, node::GradNode, ops::op_impl},
    tensor::{Element, Tensor},
};

use std::ops::{Add, Sub};

#[derive(Debug, Clone)]
pub struct AddBackward<T: Element> {
    lhs: Tensor<T>,
    rhs: Tensor<T>,
}

impl<T> BackwardFn<T> for AddBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Tensor<T>) {
        self.lhs
            .grad_node
            .get()
            .unwrap()
            .backward(grad_output.clone());
        self.rhs
            .grad_node
            .get()
            .unwrap()
            .backward(grad_output.clone());
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    fn add_grad(&self, rhs: &Self) -> Self {
        let ret = self.elemwise_add(rhs);

        let backwrd_fn = GradNode::new(Box::new(AddBackward {
            lhs: self.clone(),
            rhs: rhs.clone(),
        }));

        if self.requires_grad() || rhs.requires_grad() {
            let _ = ret.grad_node.set(backwrd_fn);
        }

        ret
    }
}

op_impl!(Add, add; add_grad, AddBackward);

#[derive(Debug, Clone)]
pub struct SubBackward<T: Element> {
    lhs: Tensor<T>,
    rhs: Tensor<T>,
}

impl<T> BackwardFn<T> for SubBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Tensor<T>) {
        self.lhs
            .grad_node
            .get()
            .unwrap()
            .backward(grad_output.clone());
        self.rhs
            .grad_node
            .get()
            .unwrap()
            .backward(Tensor::zeros_like(&grad_output, Some(false)).elemwise_sub(&grad_output));
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    fn sub_grad(&self, rhs: &Self) -> Self {
        let ret = self.elemwise_sub(rhs);

        let backwrd_fn = GradNode::new(Box::new(SubBackward {
            lhs: self.clone(),
            rhs: rhs.clone(),
        }));

        if self.requires_grad() || rhs.requires_grad() {
            let _ = ret.grad_node.set(backwrd_fn);
        }

        ret
    }
}

op_impl!(Sub, sub; sub_grad, SubBackward);

#[derive(Debug, Clone)]
pub struct MatMulBackward<T: Element> {
    lhs: Tensor<T>,
    rhs: Tensor<T>,
}

impl<T> BackwardFn<T> for MatMulBackward<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Tensor<T>) {
        self.lhs
            .grad_node
            .get()
            .unwrap()
            .backward(grad_output.forward_matmul(&self.rhs.transpose()));
        self.rhs
            .grad_node
            .get()
            .unwrap()
            .backward(self.lhs.transpose().forward_matmul(&grad_output));
    }
}

impl<T> Tensor<T>
where
    T: Element,
{
    fn matmul_grad(&self, rhs: &Self) -> Self {
        let ret = self.forward_matmul(rhs);

        let backwrd_fn = GradNode::new(Box::new(MatMulBackward {
            lhs: self.clone(),
            rhs: rhs.clone(),
        }));

        if self.requires_grad() || rhs.requires_grad() {
            let _ = ret.grad_node.set(backwrd_fn);
        }

        ret
    }
}

op_impl!(matmul; matmul_grad, MatMulBackward);
