use std::{cmp, iter::zip};

use itertools::Itertools;

use crate::tensor::metadata::TensorMetadata;

impl TensorMetadata {
    fn is_broadcastable(&self, target: &[usize]) -> bool {
        self.shape
            .iter()
            .rev()
            .zip(target.iter().rev())
            .all(|(&ss, &ts)| ss == ts || ss == 1)
            && self.shape.len() <= target.len()
    }

    pub fn broadcast_to(&self, target: &[usize]) -> Self {
        assert!(
            self.is_broadcastable(target),
            "Tensor of shape: {:?} cannot be broadcasted to shape: {:?}",
            self.shape,
            target
        );
        let mut broadcasted_stride: Vec<usize> = Vec::new();

        for dim in zip(&self.shape, &self.stride)
            .rev()
            .zip_longest(target.iter().rev())
        {
            match dim {
                itertools::EitherOrBoth::Both((self_dim, self_strd), target_dim) => {
                    match (self_dim, target_dim) {
                        (1, 1) => broadcasted_stride.push(*self_strd),
                        (1, _) => {
                            broadcasted_stride.push(0);
                        }
                        _ => broadcasted_stride.push(*self_strd),
                    }
                }
                itertools::EitherOrBoth::Left(_) => {
                    panic!("Tensor is shorter than target");
                }
                itertools::EitherOrBoth::Right(_) => {
                    broadcasted_stride.push(0);
                }
            }
        }
        broadcasted_stride.reverse();

        Self {
            stride: broadcasted_stride,
            shape: target.to_vec(),
            offset: self.offset,
        }
    }

    pub fn broadcast_tensors(&self, other: &Self) -> (Self, Self) {
        let mut new_shape: Vec<usize> = Vec::new();

        for dim in self
            .shape
            .iter()
            .rev()
            .zip_longest(other.shape.clone().iter().rev())
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
            other.shape
        );

        (
            self.broadcast_to(&new_shape),
            other.broadcast_to(&new_shape),
        )
    }
}
