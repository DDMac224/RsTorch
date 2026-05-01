use std::{cmp, sync::Arc};

use itertools::Itertools;

use crate::tensor::{Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    fn is_broadcastable(&self, target: &Vec<usize>) -> bool {
        self.shape()
            .iter()
            .rev()
            .zip(target.iter().rev())
            .all(|(&ss, &ts)| ss == ts || ss == 1)
            && self.shape.len() <= target.len()
    }

    pub fn broadcast_to(&self, target: &Vec<usize>) -> Self {
        assert!(
            self.is_broadcastable(target),
            "Tensor of shape: {:?} cannot be broadcasted to shape: {:?}",
            self.shape,
            target
        );
        let mut strd_mask: Vec<usize> = Vec::new();

        for dim in self.shape.iter().rev().zip_longest(target.iter().rev()) {
            match dim {
                itertools::EitherOrBoth::Both(self_dim, target_dim) => {
                    match (self_dim, target_dim) {
                        (1, 1) => {
                            strd_mask.push(usize::MAX);
                        }
                        (1, _) => {
                            strd_mask.push(0);
                        }
                        _ => {
                            strd_mask.push(usize::MAX);
                        }
                    }
                }
                itertools::EitherOrBoth::Left(_) => {
                    panic!("Tensor is shorter than target");
                }
                itertools::EitherOrBoth::Right(_) => {
                    strd_mask.push(0);
                }
            }
        }
        strd_mask.reverse();
        let mut self_broadcasted_stride = self.stride();

        if self.stride.len() < strd_mask.len() {
            let mut pad = vec![0; strd_mask.len() - self.stride.len()];
            pad.extend(self.stride());
            self_broadcasted_stride = pad;
        }

        self_broadcasted_stride = self_broadcasted_stride
            .iter()
            .zip(strd_mask)
            .map(|(strd, mask)| strd & mask)
            .collect();

        Self {
            data: Arc::clone(&self.data),
            stride: self_broadcasted_stride,
            shape: target.clone(),
            device: self.device(),
            offset: self.offset.clone(),
            grad: Arc::clone(&self.grad),
            requires_grad: self.requires_grad,
            // change so it doesn't need to be cloned
            grad_fn: Arc::clone(&self.grad_fn),
        }
    }

    pub fn broadcast_tensors(&self, other: &Self) -> (Self, Self) {
        let mut new_shape: Vec<usize> = Vec::new();

        for dim in self
            .shape
            .iter()
            .rev()
            .zip_longest(other.shape().iter().rev())
        {
            match dim {
                itertools::EitherOrBoth::Both(self_dim, other_dim) => {
                    new_shape.push(cmp::max(*self_dim, *other_dim));
                }
                itertools::EitherOrBoth::Left(self_dim) => {
                    new_shape.push(*self_dim);
                }
                itertools::EitherOrBoth::Right(other_dim) => {
                    new_shape.push(*other_dim);
                }
            }
        }

        new_shape.reverse();

        assert!(
            self.is_broadcastable(&new_shape) && other.is_broadcastable(&new_shape),
            "Tensor of shape: {:?} and Tensor of shape: {:?} are not broadcastable.",
            self.shape,
            other.shape()
        );

        (
            self.broadcast_to(&new_shape),
            other.broadcast_to(&new_shape),
        )
    }
}
