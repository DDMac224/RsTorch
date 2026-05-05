use std::cmp;

use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub(super) stride: Vec<usize>,
    pub(super) shape: Vec<usize>,
    pub(super) offset: usize,
}

impl TensorMetadata {
    pub fn new(shape: Vec<usize>) -> Self {
        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        Self {
            stride,
            shape,
            offset: 0,
        }
    }

    pub fn from_parts(shape: Vec<usize>, stride: Vec<usize>, offset: usize) -> Self {
        Self {
            stride,
            shape,
            offset,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.iter().product::<usize>() == 1
    }

    pub fn index(&self, idx: &[usize]) -> Self {
        assert!(
            idx.len() <= self.shape.len(),
            "Indexing too many dimensions."
        );
        assert!(
            !self.shape.iter().zip(idx.iter()).any(|(dim, i)| i >= dim),
            "Index out of bounds."
        );

        let new_offset: usize = self
            .stride
            .iter()
            .zip(idx.iter())
            .fold(0, |acc, (strd, i)| acc + strd * i)
            + self.offset;
        let new_shape = &self.shape[idx.len()..];
        let new_stride = &self.stride[idx.len()..];

        Self {
            stride: Vec::from(new_stride),
            shape: Vec::from(new_shape),
            offset: new_offset,
        }
    }

    fn is_broadcastable(&self, target: &Vec<usize>) -> bool {
        self.shape
            .clone()
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
        let mut self_broadcasted_stride = self.stride.clone();

        if self.stride.len() < strd_mask.len() {
            let mut pad = vec![0; strd_mask.len() - self.stride.len()];
            pad.extend(self.stride.clone());
            self_broadcasted_stride = pad;
        }

        self_broadcasted_stride = self_broadcasted_stride
            .iter()
            .zip(strd_mask)
            .map(|(strd, mask)| strd & mask)
            .collect();

        Self {
            stride: self_broadcasted_stride,
            shape: target.clone(),
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

    pub fn transpose(&self) -> Self {
        let mut new_shape = self.shape.clone();
        let mut new_stride = self.stride.clone();

        new_shape.swap(self.rank() - 2, self.rank() - 1);
        new_stride.swap(self.rank() - 2, self.rank() - 1);

        TensorMetadata {
            stride: new_stride,
            shape: new_shape,
            offset: self.offset.clone(),
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        let mut stride: Vec<usize> = Vec::new();
        let mut strd = 1;
        for dim in shape.iter().rev() {
            stride.push(strd);
            strd *= dim;
        }
        stride.reverse();

        TensorMetadata {
            stride,
            shape,
            offset: self.offset.clone(),
        }
    }
}
