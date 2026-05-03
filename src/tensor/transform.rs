use std::sync::{Arc, Mutex};

use crate::tensor::{Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn is_contiguous(&self) -> bool {
        let mut expt_strd: usize = 1;

        for (&strd, &shp) in self.stride.iter().rev().zip(self.shape.iter().rev()) {
            if strd != expt_strd {
                return false;
            }
            expt_strd *= shp;
        }
        return true;
    }

    pub fn contiguous(&mut self) {
        let size: usize = self.shape.iter().product();
        let mut data: Vec<T> = Vec::new();

        for i in 0..size {
            data.push(
                self.data[self.offset
                    + self
                        .shape
                        .iter()
                        .rev()
                        .scan(i, |acc, e| {
                            let temp = *acc;
                            *acc /= *e;
                            Some(temp % e)
                        })
                        .zip(self.stride.iter().rev())
                        .fold(0, |acc, (idx, strd)| acc + (idx * strd))],
            );
        }

        self.data = Arc::new(data);
        self.offset = 0;

        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in self.shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        self.stride = stride;
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
            let mut cont = self.clone();
            cont.contiguous();
            data = cont.data;
        }
        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        Self {
            data: data,
            stride: stride,
            shape: shape,
            device: self.device(),
            offset: self.offset,
            grad_node: self.grad_node.clone(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut data = self.data.clone();
        if !self.is_contiguous() {
            let mut cont = self.clone();
            cont.contiguous();
            data = cont.data;
        }

        let mut new_shape = self.shape();
        let mut new_stride = self.stride();
        new_shape.swap(self.shape.len() - 2, self.shape.len() - 1);
        new_stride.swap(self.stride.len() - 2, self.stride.len() - 1);

        Self {
            data: data,
            stride: new_stride,
            shape: new_shape,
            device: self.device(),
            offset: self.offset,
            grad_node: self.grad_node.clone(),
            requires_grad: self.requires_grad,
        }
    }
}
