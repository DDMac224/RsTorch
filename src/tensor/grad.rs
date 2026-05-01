use std::sync::Arc;

use crate::{
    autograd::BackwardFn,
    tensor::{Element, Tensor},
};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub(crate) fn set_grad_fn(&self, grad_fn: Arc<dyn BackwardFn<T>>) {
        let mut mutable_fn = self.grad_fn.lock().unwrap();
        match *mutable_fn {
            Some(ref mut fn_ref) => *fn_ref = Arc::clone(&grad_fn),
            None => *mutable_fn = Some(Arc::clone(&grad_fn)),
        }
    }

    pub(crate) fn set_grad(&self, grad: Arc<Tensor<T>>) {
        let mut mutable_grad = self.grad.lock().unwrap();
        match *mutable_grad {
            Some(ref mut grad_ref) => *grad_ref = Arc::clone(&grad),
            None => *mutable_grad = Some(Arc::clone(&grad)),
        }
    }

    pub(crate) fn update_grad(&self, grad: Arc<Tensor<T>>) {
        let mut mutable_grad = self.grad.lock().unwrap();
        match *mutable_grad {
            Some(ref mut grad_ref) => *grad_ref = Arc::clone(&grad).elemwise_add(grad_ref).into(),
            None => *mutable_grad = Some(Arc::clone(&grad)),
        }
    }
}

impl<T> BackwardFn<T> for Tensor<T>
where
    T: Element,
{
    fn backward(&self, grad_output: Arc<Tensor<T>>) {
        self.update_grad(Arc::clone(&grad_output));
        match *self.grad_fn.lock().unwrap() {
            Some(ref fn_ref) => fn_ref.backward(Arc::clone(&grad_output)),
            None => (),
        }
    }

    fn zero_grad(&self) {
        let zero = Arc::new(Tensor::zeros_like(&self, None));
        self.set_grad(Arc::clone(&zero));
        match *self.grad_fn.lock().unwrap() {
            Some(ref fn_ref) => fn_ref.zero_grad(),
            None => (),
        }
    }
}
