use std::{cmp, iter::repeat_n};

use itertools::Itertools;

use crate::tensor::{Device, Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn cpu_elemwise(&self, rhs: &Self, op: fn(T, T) -> T) -> Self {
        let mut brdcsted_self = self.clone();
        let mut brdcsted_rhs = rhs.clone();
        if self.shape != rhs.shape {
            (brdcsted_self, brdcsted_rhs) = self.broadcast_tensors(rhs);
        }

        let size: usize = brdcsted_self.shape.iter().product();

        let mut data: Vec<T> = Vec::new();

        for i in 0..size {
            let idx = brdcsted_self.shape.iter().rev().scan(i, |acc, e| {
                let temp = *acc;
                *acc /= *e;
                Some(temp % e)
            });
            let elem_self = brdcsted_self.offset
                + idx
                    .clone()
                    .zip(brdcsted_self.stride.iter())
                    .fold(0, |acc, (idx, strd)| acc + (idx * strd));
            let elem_rhs = brdcsted_rhs.offset
                + idx
                    .zip(brdcsted_rhs.stride.iter())
                    .fold(0, |acc, (idx, strd)| acc + (idx * strd));

            data.push(op(
                brdcsted_self.data[elem_self],
                brdcsted_rhs.data[elem_rhs],
            ));
        }

        Tensor::new(data, brdcsted_self.shape(), Some(Device::CPU))
    }

    fn matmul_matricies(&self, rhs: &Self) -> Self {
        assert!(
            self.shape.len() == rhs.shape().len() && self.shape.len() == 2,
            "Matrix matmul can only have two dimensions"
        );
        assert_eq!(
            self.shape[1],
            rhs.shape()[0],
            "Columns of self and rows of rhs must match"
        );

        let new_dims = vec![self.shape[0], rhs.shape()[1]];

        let self_rows = self.data.iter().skip(self.offset).step_by(self.stride[1]);
        let rhs_cols = rhs.data.iter().skip(rhs.offset).step_by(rhs.stride[0]);

        let products = self_rows.zip(rhs_cols).map(|(&s, &r)| s * r);
        let new_data = products
            .chunks(self.shape[1])
            .into_iter()
            .map(|chunk| chunk.reduce(|acc, e| acc + e).unwrap())
            .collect();

        Tensor::new(new_data, new_dims, Some(self.device()))
    }

    fn batched_matmul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.shape[self.shape.len() - 1],
            rhs.shape()[rhs.shape.len()],
            "Columns of self and rows of rhs must match"
        );

        assert!(
            self.shape.len() > 2 || rhs.shape.len() > 2,
            "One tensor must have more than 2 dimensions."
        );

        let mut brdcsted_self = self.clone();
        let mut brdcsted_rhs = rhs.clone();
        if self.shape[self.shape.len() - 2..] != rhs.shape[rhs.shape.len() - 2..] {
            let mut new_shape: Vec<usize> = Vec::new();

            for dim in self
                .shape
                .iter()
                .rev()
                .zip_longest(rhs.shape().iter().rev())
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
            new_shape.truncate(new_shape.len() - 2);

            let mut new_shape_self = new_shape.clone();
            new_shape_self.extend_from_slice(&self.shape()[self.shape.len() - 2..]);
            let mut new_shape_rhs = new_shape;
            new_shape_rhs.extend_from_slice(&rhs.shape()[rhs.shape.len() - 2..]);

            brdcsted_self = self.broadcast_to(&new_shape_self);
            brdcsted_rhs = rhs.broadcast_to(&new_shape_rhs);
        }

        let mut new_shape = brdcsted_self.shape();
        new_shape.truncate(brdcsted_self.shape.len() - 2);
        let size: usize = new_shape.iter().product();
        new_shape.push(brdcsted_self.shape[brdcsted_self.shape.len() - 1]);
        new_shape.push(brdcsted_rhs.shape[brdcsted_rhs.shape.len()]);

        let mut new_data: Vec<T> = Vec::new();

        for i in 0..size {
            let offset_idx = repeat_n(0, 2)
                .chain(new_shape.iter().rev().scan(i, |acc, e| {
                    let temp = *acc;
                    *acc /= e;
                    Some(e % temp)
                }))
                .collect::<Vec<usize>>();
            let elem_self = brdcsted_self.index(offset_idx.clone());
            let elem_rhs = brdcsted_rhs.index(offset_idx);

            new_data.extend_from_slice(elem_self.matmul_matricies(&elem_rhs).data.as_slice());
        }

        Tensor::new(new_data, new_shape, Some(Device::CPU))
    }

    pub fn cpu_matmul(&self, rhs: &Self) -> Self {
        if self.shape.len() == 2 && rhs.shape.len() == 2 {
            return self.matmul_matricies(rhs);
        } else if self.is_scalar() || rhs.is_scalar() {
            return self.cpu_elemwise(rhs, T::mul);
        } else {
            return self.batched_matmul(rhs);
        }
    }
}
