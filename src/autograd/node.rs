use std::sync::Mutex;

use crate::{
    autograd::BackwardFn,
    tensor::{Element, Tensor},
};

#[derive(Debug)]
pub(crate) struct GradNode<T: Element> {
    pub(crate) grad: Mutex<Option<Tensor<T>>>,
    pub(crate) grad_fn: Option<Box<dyn BackwardFn<T>>>,
}

impl<T: Element> Clone for GradNode<T> {
    fn clone(&self) -> Self {
        GradNode {
            grad: Mutex::new(None),
            grad_fn: self.grad_fn.clone(),
        }
    }
}

impl<T: Element> GradNode<T> {
    pub fn leaf() -> Self {
        Self {
            grad: Mutex::new(None),
            grad_fn: None,
        }
    }

    pub fn new(grad_fn: Box<dyn BackwardFn<T>>) -> Self {
        Self {
            grad: Mutex::new(None),
            grad_fn: Some(grad_fn),
        }
    }

    pub fn set_grad(&self, grad: Tensor<T>) {
        *self.grad.lock().unwrap() = Some(grad);
    }

    pub fn update_grad(&self, grad: Tensor<T>) {
        let mut grad_mut = self.grad.lock().unwrap();

        if let Some(ref mut mut_t) = *grad_mut {
            *mut_t = mut_t.elemwise_add(&grad);
        } else {
            *grad_mut = Some(grad);
        }
    }

    pub fn accumulate_grad(&self, seed: &Tensor<T>) {
        match self.grad_fn {
            Some(ref fun) => fun.backward(seed.clone()),
            None => panic!("Leaf node cannot call backward()"),
        }
    }

    pub fn backward(&self, grad_output: Tensor<T>) {
        if let Some(ref backfn) = self.grad_fn {
            backfn.backward(grad_output);
        } else {
            self.update_grad(grad_output);
        }
    }
}
