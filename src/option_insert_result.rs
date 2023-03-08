// Similar to Option<T>.get_or_insert_with, for a function that returns a result.
pub trait OptionInsertWithResult<T> {
    fn get_or_insert_with_result<F, E>(&mut self, f: F) -> Result<&mut T, E>
    where
        F: FnOnce() -> Result<T, E>;
}

impl<T> OptionInsertWithResult<T> for Option<T> {
    fn get_or_insert_with_result<F, E>(&mut self, f: F) -> Result<&mut T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if self.is_none() {
            *self = Some(f()?);
        }
        Ok(self.as_mut().unwrap())
    }
}
