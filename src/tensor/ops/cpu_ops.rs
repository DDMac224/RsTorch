use std::{iter::repeat_n, ops::Add, sync::Arc};

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
                Some(e % temp)
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

        let mut brdcsted_self = self.clone();
        let mut brdcsted_rhs = rhs.clone();
        if self.shape != rhs.shape {
            (brdcsted_self, brdcsted_rhs) = self.broadcast_tensors(rhs);
        }

        let mut sliced_self_shape = brdcsted_self.shape();
        sliced_self_shape.truncate(brdcsted_self.shape.len() - 2);
        let size: usize = sliced_self_shape.iter().product();

        for i in 0..size {
            let offset_idx =
                repeat_n(0, 2).chain(sliced_self_shape.iter().rev().scan(i, |acc, e| {
                    let temp = *acc;
                    *acc /= e;
                    Some(e % temp)
                }));
            let elem_self = brdcsted_self.offset
                + offset_idx
                    .clone()
                    .zip(brdcsted_self.stride.iter())
                    .fold(0, |acc, (idx, strd)| acc + (idx * strd));
            let elem_rhs = brdcsted_rhs.offset
                + offset_idx
                    .zip(brdcsted_rhs.stride.iter())
                    .fold(0, |acc, (idx, strd)| acc + (idx * strd));
        }

        todo!()
    }
}
