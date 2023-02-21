// Similar to Option<T>.get_or_insert_with, for a function that returns a result.
pub fn get_or_insert_with_result<T, F, E>(opt: &mut Option<T>, f: F) -> Result<&T, E>
    where F: FnOnce() -> Result<T, E>
{
    if opt.is_none() {
        *opt = Some(f()?);
    }
    Ok(opt.as_ref().unwrap())
}
