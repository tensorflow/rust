#[doc(hidden)]
macro_rules! link {
    (
        $(
            extern "C" {
                pub fn $name:ident($($pname:ident: $pty:ty),* $(,)?) $(-> $ret:ty)*;
            }
        )+
    ) => (
        use lazy_static::lazy_static;
        use libloading;
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::sync::RwLock;

        // Wrap the loaded functions.
        pub(crate) struct SharedLibrary {
            library: libloading::Library,
            pub functions: Functions,
        }
        impl SharedLibrary {
            fn new(library: libloading::Library) -> Self {
                Self {
                    library,
                    functions: Functions::default(),
                }
            }
        }

        // `LIBRARY` holds the shared library reference.
        lazy_static!{
            static ref LIBRARY: RwLock<Option<Arc<SharedLibrary>>> = RwLock::new(None);
        }

        // Helper function for accessing the thread-local version of the library.
        fn with_library<T, F>(f: F) -> Option<T>
        where
            F: FnOnce(&SharedLibrary) -> T,
        {
            match LIBRARY.read().unwrap().as_ref() {
                Some(library) => Some(f(&library)),
                _ => None,
            }
        }

        // The set of functions loaded dynamically.
        #[derive(Default)]
        pub(crate) struct Functions {
            $(
                pub $name: Option<unsafe extern fn($($pname: $pty), *) $(-> $ret)*>,
            )+
        }

        // Provide functions to load each name from the shared library into the `SharedLibrary`
        // struct.
        mod load {
            $(
                pub(crate) fn $name(library: &mut super::SharedLibrary) {
                    let symbol = unsafe { library.library.get(stringify!($name).as_bytes()) }.ok();
                    library.functions.$name = match symbol {
                        Some(s) => *s,
                        None => None,
                    };
                }
            )+
        }

        /// Load all of the function definitions from a shared library.
        pub fn load() -> Result<(), String> {
            match crate::library::find() {
                None => Err("Unable to find the `libtensorflow` library to load".into()),
                Some(path) => load_from(path),
            }
        }
        fn load_from(path: PathBuf) -> Result<(), String> {
            let library = Arc::new(SharedLibrary::load(path)?);

            let mut static_lib = LIBRARY.write().map_err(|e| {
                format!( "Failed to acquire global lock for TensorFlow library {}",e)
            })?;

            *static_lib = Some(library);
            Ok(())
        }
        impl SharedLibrary {
            fn load(path: PathBuf) -> Result<SharedLibrary, String> {
                unsafe {
                    let library = libloading::Library::new(&path).map_err(|e| {
                        format!(
                            "the shared library at {} could not be opened: {}",
                            path.display(),
                            e,
                        )
                    });

                    let mut library = SharedLibrary::new(library?);
                    $(load::$name(&mut library);)+
                    Ok(library)
                }
            }
        }
        // For each loaded function, we redefine them to proxy their call through the SharedLibrary
        // on the local thread and into the loaded shared library implementation.
        $(
            pub unsafe fn $name($($pname: $pty), *) $(-> $ret)* {
                let f = with_library(|l| {
                    l.functions.$name.expect(concat!(
                        "`libtensorflow` function not loaded: `",
                        stringify!($name)
                    ))
                }).expect("an `libtensorflow` shared library is not loaded on this thread");
                f($($pname), *)
            }
        )+
    )
}
