pub mod broadcast;

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

    pub fn expand_idx(&self, idx: usize) -> Vec<usize> {
        let mut ret: Vec<usize> = self
            .shape
            .iter()
            .rev()
            .scan(idx, |acc, e| {
                let temp = *acc;
                *acc /= *e;
                Some(temp % e)
            })
            .collect();
        ret.reverse();
        ret
    }

    pub fn transpose(&self) -> Self {
        let mut new_shape = self.shape.clone();
        let mut new_stride = self.stride.clone();

        new_shape.swap(self.rank() - 2, self.rank() - 1);
        new_stride.swap(self.rank() - 2, self.rank() - 1);

        TensorMetadata {
            stride: new_stride,
            shape: new_shape,
            offset: self.offset,
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
            offset: self.offset,
        }
    }
}
