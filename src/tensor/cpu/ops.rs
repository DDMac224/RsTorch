use std::{cmp, iter::repeat_n};

use itertools::Itertools;

use crate::tensor::{Device, Element, Tensor};

impl<T> Tensor<T>
where
    T: Element,
{
    pub fn cpu_elemwise_bin(&self, rhs: &Self, op: fn(T, T) -> T) -> Self {
        let mut brdcsted_self = self.clone();
        let mut brdcsted_rhs = rhs.clone();
        if self.metadata.shape != rhs.metadata.shape {
            (brdcsted_self, brdcsted_rhs) = self.broadcast_tensors(rhs);
        }

        let size: usize = brdcsted_self.metadata.shape.iter().product();

        let mut data: Vec<T> = Vec::new();

        let self_data_locked = self.data.read().unwrap();
        let rhs_data_locked = rhs.data.read().unwrap();

        let self_cpu_data = self_data_locked.expect_cpu();
        let rhs_cpu_data = rhs_data_locked.expect_cpu();

        for i in 0..size {
            let idx = brdcsted_self.metadata.shape.iter().rev().scan(i, |acc, e| {
                let temp = *acc;
                *acc /= *e;
                Some(temp % e)
            });
            let elem_self = brdcsted_self.metadata.offset
                + idx
                    .clone()
                    .zip(brdcsted_self.metadata.stride.iter().rev())
                    .fold(0, |acc, (idx, strd)| acc + (idx * strd));
            let elem_rhs = brdcsted_rhs.metadata.offset
                + idx
                    .zip(brdcsted_rhs.metadata.stride.iter().rev())
                    .fold(0, |acc, (idx, strd)| acc + (idx * strd));

            data.push(op(self_cpu_data[elem_self], rhs_cpu_data[elem_rhs]));
        }

        Tensor::new_from_op(data, brdcsted_self.shape(), Device::CPU)
    }

    pub fn cpu_elemwise_uni(&self, op: fn(T) -> T) -> Self {
        let size = self.size();

        let mut data: Vec<T> = Vec::new();

        let self_data_locked = self.data.read().unwrap();
        let self_cpu_data = self_data_locked.expect_cpu();

        if self.is_contiguous() {
            for i in 0..size {
                data.push(op(self_cpu_data[self.metadata.offset + i]));
            }
        } else {
            for i in 0..size {
                let idx = self.metadata.shape.iter().rev().scan(i, |acc, e| {
                    let temp = *acc;
                    *acc /= *e;
                    Some(temp % e)
                });
                let elem_self = self.metadata.offset
                    + idx
                        .clone()
                        .zip(self.metadata.stride.iter().rev())
                        .fold(0, |acc, (idx, strd)| acc + (idx * strd));

                data.push(op(self_cpu_data[elem_self]));
            }
        }

        Tensor::new_from_op(data, self.shape(), Device::CPU)
    }

    fn matmul_matricies(&self, rhs: &Self) -> Self {
        assert!(
            self.rank() == rhs.rank() && self.rank() == 2,
            "Matrix matmul can only have two dimensions"
        );
        assert_eq!(
            self.metadata.shape[1], rhs.metadata.shape[0],
            "Columns of self and rows of rhs must match"
        );

        let new_dims = vec![self.metadata.shape[0], rhs.shape()[1]];

        let mut data: Vec<T> = Vec::new();

        let self_data_locked = self.data.read().unwrap();
        let rhs_data_locked = rhs.data.read().unwrap();

        let self_cpu_data = self_data_locked.expect_cpu();
        let rhs_cpu_data = rhs_data_locked.expect_cpu();

        for i in 0..self.metadata.shape[0] {
            for j in 0..rhs.metadata.shape[1] {
                let sum = (0..self.metadata.shape[1])
                    .map(|k| {
                        let self_elem = self_cpu_data[self.metadata.offset
                            + i * self.metadata.stride[0]
                            + k * self.metadata.stride[1]];
                        let rhs_elem = rhs_cpu_data[rhs.metadata.offset
                            + j * rhs.metadata.stride[1]
                            + k * rhs.metadata.stride[0]];
                        self_elem * rhs_elem
                    })
                    .reduce(|acc, e| acc + e)
                    .expect("Data was empty");

                data.push(sum);
            }
        }

        Tensor::new_from_op(data, new_dims, self.device())
    }

    fn batched_matmul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.metadata.shape[self.metadata.shape.len() - 1],
            rhs.shape()[rhs.metadata.shape.len()],
            "Columns of self and rows of rhs must match"
        );

        assert!(
            self.rank() > 2 || rhs.rank() > 2,
            "One tensor must have more than 2 dimensions."
        );

        let mut brdcsted_self = self.clone();
        let mut brdcsted_rhs = rhs.clone();
        if self.metadata.shape[self.rank() - 2..] != rhs.metadata.shape[rhs.rank() - 2..] {
            let mut new_shape: Vec<usize> = Vec::new();

            for dim in self
                .metadata
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
            new_shape_self.extend_from_slice(&self.shape()[self.rank() - 2..]);
            let mut new_shape_rhs = new_shape;
            new_shape_rhs.extend_from_slice(&rhs.shape()[rhs.rank() - 2..]);

            brdcsted_self = self.broadcast_to(&new_shape_self);
            brdcsted_rhs = rhs.broadcast_to(&new_shape_rhs);
        }

        let mut new_shape = brdcsted_self.shape();
        new_shape.truncate(brdcsted_self.rank() - 2);
        let size: usize = new_shape.iter().product();
        new_shape.push(brdcsted_self.metadata.shape[brdcsted_self.rank() - 1]);
        new_shape.push(brdcsted_rhs.metadata.shape[brdcsted_rhs.rank()]);

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

            new_data.extend_from_slice(
                elem_self
                    .matmul_matricies(&elem_rhs)
                    .data
                    .read()
                    .unwrap()
                    .expect_cpu()
                    .as_slice(),
            );
        }

        Tensor::new_from_op(new_data, new_shape, Device::CPU)
    }

    pub fn cpu_matmul(&self, rhs: &Self) -> Self {
        if self.rank() == 2 && rhs.rank() == 2 {
            return self.matmul_matricies(rhs);
        } else if self.metadata.is_scalar() || rhs.metadata.is_scalar() {
            return self.cpu_elemwise_bin(rhs, T::mul);
        } else {
            return self.batched_matmul(rhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor};

    fn collect<T: crate::tensor::Element + Default>(t: &Tensor<T>) -> Vec<T> {
        let shape = t.shape();
        let size: usize = shape.iter().product();
        let ndim = shape.len();

        (0..size)
            .map(|flat| {
                let mut idx = vec![0usize; ndim];
                let mut rem = flat;
                for d in (0..ndim).rev() {
                    idx[d] = rem % shape[d];
                    rem /= shape[d];
                }
                t.index(&idx).item().unwrap()
            })
            .collect()
    }

    fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::new(data, shape, Some(Device::CPU), None)
    }

    // =========================================================================
    // cpu_elemwise_bin — broadcasting (private broadcast internals)
    // =========================================================================

    mod elemwise_bin {
        use super::*;

        mod broadcasting {
            use super::*;

            #[test]
            fn scalar_rhs() {
                let a = t(vec![1.0, 2.0, 3.0], vec![3]);
                let b = t(vec![10.0], vec![1]);
                let c = a.cpu_elemwise_bin(&b, |x, y| x + y);
                assert_eq!(c.shape(), vec![3]);
                assert_eq!(collect(&c), vec![11.0, 12.0, 13.0]);
            }

            #[test]
            fn scalar_lhs() {
                let a = t(vec![10.0], vec![1]);
                let b = t(vec![1.0, 2.0, 3.0], vec![3]);
                let c = a.cpu_elemwise_bin(&b, |x, y| x + y);
                assert_eq!(collect(&c), vec![11.0, 12.0, 13.0]);
            }

            #[test]
            fn row_vector_to_matrix() {
                let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
                let b = t(vec![10.0, 20.0], vec![1, 2]);
                let c = a.cpu_elemwise_bin(&b, |x, y| x + y);
                assert_eq!(collect(&c), vec![11.0, 22.0, 13.0, 24.0]);
            }

            #[test]
            fn col_vector_to_matrix() {
                let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
                let b = t(vec![10.0, 20.0], vec![2, 1]);
                let c = a.cpu_elemwise_bin(&b, |x, y| x + y);
                assert_eq!(collect(&c), vec![11.0, 12.0, 23.0, 24.0]);
            }

            #[test]
            fn broadcast_3d() {
                let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]);
                let b = t(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![1, 2, 3]);
                let c = a.cpu_elemwise_bin(&b, |x, y| x + y);
                assert_eq!(collect(&c), vec![
                    11.0, 22.0, 33.0, 41.0, 52.0, 63.0,
                    14.0, 25.0, 36.0, 44.0, 55.0, 66.0,
                ]);
            }
        }

        mod custom_ops {
            use super::*;

            #[test]
            fn arbitrary_closure_max() {
                let a = t(vec![1.0, 5.0, 3.0], vec![3]);
                let b = t(vec![4.0, 2.0, 3.0], vec![3]);
                let c = a.cpu_elemwise_bin(&b, f32::max);
                assert_eq!(collect(&c), vec![4.0, 5.0, 3.0]);
            }

            #[test]
            fn arbitrary_closure_abs_diff() {
                let a = t(vec![1.0, 10.0, 3.0], vec![3]);
                let b = t(vec![4.0, 2.0, 8.0], vec![3]);
                let c = a.cpu_elemwise_bin(&b, |x, y| (x - y).abs());
                assert_eq!(collect(&c), vec![3.0, 8.0, 5.0]);
            }
        }

        mod output_properties {
            use super::*;

            #[test]
            fn result_device_is_cpu() {
                let a = t(vec![1.0], vec![1]);
                let b = t(vec![2.0], vec![1]);
                assert_eq!(a.cpu_elemwise_bin(&b, |x, y| x + y).device(), Device::CPU);
            }

            #[test]
            fn result_shape_matches_broadcast_shape() {
                let a = t(vec![1.0, 2.0, 3.0], vec![1, 3]);
                let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
                assert_eq!(a.cpu_elemwise_bin(&b, |x, y| x + y).shape(), vec![2, 3]);
            }
        }
    }

    // =========================================================================
    // cpu_elemwise_uni — contiguous and non-contiguous code paths
    // =========================================================================

    mod elemwise_uni {
        use super::*;

        mod contiguous {
            use super::*;

            #[test]
            fn square_contiguous_1d() {
                let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
                assert!(a.is_contiguous());
                let c = a.cpu_elemwise_uni(|x| x * x);
                assert_eq!(collect(&c), vec![1.0, 4.0, 9.0, 16.0]);
            }

            #[test]
            fn square_contiguous_2d() {
                let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
                assert!(a.is_contiguous());
                let c = a.cpu_elemwise_uni(|x| x + 10.0);
                assert_eq!(collect(&c), vec![11.0, 12.0, 13.0, 14.0]);
            }
        }

        mod non_contiguous {
            use super::*;

            #[test]
            fn square_broadcast_row_vector() {
                let a = t(vec![1.0, 2.0, 3.0], vec![1, 3]);
                let b = a.broadcast_to(&vec![2, 3]);
                assert!(!b.is_contiguous());
                let c = b.cpu_elemwise_uni(|x| x * 2.0);
                assert_eq!(collect(&c), vec![2.0, 4.0, 6.0, 2.0, 4.0, 6.0]);
            }

            #[test]
            fn square_broadcast_column_vector() {
                let a = t(vec![1.0, 2.0], vec![2, 1]);
                let b = a.broadcast_to(&vec![2, 3]);
                assert!(!b.is_contiguous());
                let c = b.cpu_elemwise_uni(|x| x + 1.0);
                assert_eq!(collect(&c), vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
            }
        }
    }

    // =========================================================================
    // cpu_matmul — 2D path (matmul_matricies)
    // =========================================================================

    mod matmul_2d {
        use super::*;

        #[test]
        fn square_2x2() {
            let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let b = t(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
            let c = a.cpu_matmul(&b);
            assert_eq!(collect(&c), vec![19.0, 22.0, 43.0, 50.0]);
        }

        #[test]
        fn rectangular_2x3_times_3x2() {
            let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
            let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
            let c = a.cpu_matmul(&b);
            assert_eq!(collect(&c), vec![58.0, 64.0, 139.0, 154.0]);
        }

        #[test]
        fn dot_product_1x3_times_3x1() {
            let a = t(vec![1.0, 2.0, 3.0], vec![1, 3]);
            let b = t(vec![4.0, 5.0, 6.0], vec![3, 1]);
            let c = a.cpu_matmul(&b);
            assert_eq!(c.item().unwrap(), 32.0);
        }

        #[test]
        fn outer_product_3x1_times_1x3() {
            let a = t(vec![1.0, 2.0, 3.0], vec![3, 1]);
            let b = t(vec![4.0, 5.0, 6.0], vec![1, 3]);
            let c = a.cpu_matmul(&b);
            assert_eq!(collect(&c), vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);
        }

        #[test]
        fn identity_matrix() {
            let eye = t(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
            let a = t(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
            assert_eq!(collect(&a.cpu_matmul(&eye)), vec![3.0, 4.0, 5.0, 6.0]);
        }

        #[test]
        fn zero_matrix() {
            let zeros = t(vec![0.0; 4], vec![2, 2]);
            let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            assert_eq!(collect(&a.cpu_matmul(&zeros)), vec![0.0; 4]);
        }

        #[test]
        fn result_device_is_cpu() {
            let a = t(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
            let b = t(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
            assert_eq!(a.cpu_matmul(&b).device(), Device::CPU);
        }

        #[test]
        #[should_panic]
        fn mismatched_inner_dims_panics() {
            let a = t(vec![1.0, 2.0, 3.0], vec![1, 3]);
            let b = t(vec![1.0, 2.0], vec![2, 1]);
            let _ = a.cpu_matmul(&b);
        }
    }

    // =========================================================================
    // cpu_matmul — scalar fast-path
    // =========================================================================

    mod matmul_scalar {
        use super::*;

        #[test]
        fn scalar_lhs_broadcasts_over_1d() {
            let scalar = t(vec![3.0], vec![1]);
            let a = t(vec![1.0, 2.0, 3.0], vec![3]);
            assert_eq!(collect(&scalar.cpu_matmul(&a)), vec![3.0, 6.0, 9.0]);
        }

        #[test]
        fn scalar_rhs_broadcasts_over_1d() {
            let a = t(vec![1.0, 2.0, 3.0], vec![3]);
            let scalar = t(vec![2.0], vec![1]);
            assert_eq!(collect(&a.cpu_matmul(&scalar)), vec![2.0, 4.0, 6.0]);
        }

        #[test]
        fn scalar_times_scalar() {
            let a = t(vec![3.0], vec![1]);
            let b = t(vec![4.0], vec![1]);
            let c = a.cpu_matmul(&b);
            assert!(c.metadata.is_scalar());
            assert_eq!(c.item().unwrap(), 12.0);
        }

        #[test]
        fn scalar_times_2d_matrix() {
            let scalar = t(vec![2.0], vec![1]);
            let m = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let c = scalar.cpu_matmul(&m);
            assert_eq!(collect(&c), vec![2.0, 4.0, 6.0, 8.0]);
        }
    }

    // =========================================================================
    // cpu_matmul — batched path (batched_matmul)
    //
    // NOTE: These tests document expected behavior. The implementation
    // currently has bugs — see known issues at the top of each section.
    // =========================================================================

    mod matmul_batched {
        use super::*;

        mod batched_3d {
            use super::*;

            // Known bugs in batched_matmul:
            //   1. Line 127: rhs.shape()[rhs.metadata.shape.len()] is OOB
            //   2. Lines 174-175: output shape construction is incorrect
            //   3. Lines 180-184: batch index iteration logic is broken

            #[test]
            fn two_batches_2x3_times_3x2() {
                // Batch 0: [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
                // Batch 1: [[7,8,9],[10,11,12]] @ [[13,14],[15,16],[17,18]] = [[364,388],[499,532]]
                let a = t(
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    vec![2, 2, 3],
                );
                let b = t(
                    vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                    vec![2, 3, 2],
                );
                let c = a.cpu_matmul(&b);
                assert_eq!(c.shape(), vec![2, 2, 2]);
                assert_eq!(
                    collect(&c),
                    vec![58.0, 64.0, 139.0, 154.0, 364.0, 388.0, 499.0, 532.0]
                );
            }

            #[test]
            fn three_batches_2x2_times_2x2() {
                let a = t(
                    vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0],
                    vec![3, 2, 2],
                );
                let b = t(
                    vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    vec![3, 2, 2],
                );
                let c = a.cpu_matmul(&b);
                assert_eq!(c.shape(), vec![3, 2, 2]);
                assert_eq!(collect(&c), vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
            }

            #[test]
            fn result_device_is_cpu() {
                let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 1, 3]);
                let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
                assert_eq!(a.cpu_matmul(&b).device(), Device::CPU);
            }
        }

        mod batched_broadcast {
            use super::*;

            #[test]
            fn broadcast_single_matrix_across_batch() {
                // a: [2, 3, 2], b: [1, 2, 1] → b broadcasts to [2, 2, 1]
                // Result: [2, 3, 1]
                // Batch 0: [[1,2],[3,4],[5,6]] @ [[1],[2]] = [[5],[11],[17]]
                // Batch 1: [[7,8],[9,10],[11,12]] @ [[1],[2]] = [[23],[29],[35]]
                let a = t(
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    vec![2, 3, 2],
                );
                let b = t(vec![1.0, 2.0], vec![1, 2, 1]);
                let c = a.cpu_matmul(&b);
                assert_eq!(c.shape(), vec![2, 3, 1]);
                assert_eq!(collect(&c), vec![5.0, 11.0, 17.0, 23.0, 29.0, 35.0]);
            }

            #[test]
            fn batch_dim_no_broadcast() {
                // a: [2, 1, 2], b: [2, 2, 1] — same batch dim, no broadcast
                // Batch 0: [[1,2]] @ [[5],[7]] = [19]
                // Batch 1: [[3,4]] @ [[6],[8]] = [50]
                let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
                let b = t(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]);
                let c = a.cpu_matmul(&b);
                assert_eq!(c.shape(), vec![2, 1, 1]);
                let v = collect(&c);
                assert!((v[0] - 19.0).abs() < 1e-5, "batch 0: {} != 19", v[0]);
                assert!((v[1] - 50.0).abs() < 1e-5, "batch 1: {} != 50", v[1]);
            }

            #[test]
            fn broadcast_identity_matrix() {
                // a: [3, 2, 2], b: [1, 2, 2] → identity broadcasts to all 3 batches
                let a = t(
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    vec![3, 2, 2],
                );
                let b = t(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]);
                let c = a.cpu_matmul(&b);
                assert_eq!(c.shape(), vec![3, 2, 2]);
                assert_eq!(
                    collect(&c),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                );
            }
        }

        mod batched_4d {
            use super::*;

            #[test]
            fn two_by_two_batches() {
                // a: [2, 2, 2, 3], b: [2, 2, 3, 2] → result [2, 2, 2, 2]
                let a = Tensor::new(
                    (1..=24).map(|x| x as f32).collect(),
                    vec![2, 2, 2, 3],
                    Some(Device::CPU),
                    None,
                );
                let b = Tensor::new(
                    (1..=24).map(|x| x as f32).collect(),
                    vec![2, 2, 3, 2],
                    Some(Device::CPU),
                    None,
                );
                let c = a.cpu_matmul(&b);
                assert_eq!(c.shape(), vec![2, 2, 2, 2]);
                assert_eq!(c.device(), Device::CPU);
            }
        }
    }
}
