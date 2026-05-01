mod add;

macro_rules! op_impl {
    ($Trait:ident, $method:ident; $fn:ident, $backward:ident) => {
        impl<T> $Trait<&Tensor<T>> for &Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                assert_eq!(
                    self.device(),
                    rhs.device(),
                    "Tensors must be on the same device"
                );

                self.$fn(rhs)
            }
        }

        impl<T> $Trait<Tensor<T>> for Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        impl<T> $Trait<&Tensor<T>> for Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl<T> $Trait<Tensor<T>> for &Tensor<T>
        where
            T: Element,
        {
            type Output = Tensor<T>;
            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

pub(crate) use op_impl;
