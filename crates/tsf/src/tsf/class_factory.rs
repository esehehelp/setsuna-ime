//! IClassFactory for the Setsuna text service.
//!
//! Pattern taken from karukan-tsf (MIT/Apache-2.0). The only allocation
//! path the DLL ever hands out is `SetsunaTextService` — no aggregation.

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::System::Com::*;

use crate::globals::{dll_add_ref, dll_release};
use crate::tsf::text_input_processor::SetsunaTextService;

#[implement(IClassFactory)]
pub struct SetsunaClassFactory;

#[allow(clippy::new_without_default)]
impl SetsunaClassFactory {
    pub fn new() -> Self {
        dll_add_ref();
        Self
    }
}

impl Drop for SetsunaClassFactory {
    fn drop(&mut self) {
        dll_release();
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl IClassFactory_Impl for SetsunaClassFactory_Impl {
    fn CreateInstance(
        &self,
        punkouter: Option<&IUnknown>,
        riid: *const GUID,
        ppvobject: *mut *mut core::ffi::c_void,
    ) -> Result<()> {
        setsuna_core::diagnostics::log("ClassFactory::CreateInstance");
        unsafe {
            if ppvobject.is_null() {
                setsuna_core::diagnostics::log("ClassFactory::CreateInstance E_POINTER");
                return Err(E_POINTER.into());
            }
            *ppvobject = core::ptr::null_mut();
            if punkouter.is_some() {
                setsuna_core::diagnostics::log(
                    "ClassFactory::CreateInstance CLASS_E_NOAGGREGATION",
                );
                return Err(CLASS_E_NOAGGREGATION.into());
            }
            // the service's state is made here (the bridge with its
            // defaults): a panic in it never reaches the host
            let Some(service) = crate::guard::run("CreateInstance", || {
                IUnknown::from(SetsunaTextService::new())
            }) else {
                return Err(E_FAIL.into());
            };
            let result = service.query(&*riid, ppvobject).ok();
            setsuna_core::diagnostics::log(format!(
                "ClassFactory::CreateInstance result={result:?}"
            ));
            result
        }
    }

    fn LockServer(&self, flock: BOOL) -> Result<()> {
        if flock.as_bool() {
            dll_add_ref();
        } else {
            dll_release();
        }
        Ok(())
    }
}
