//! `ITfDisplayAttributeProvider` + `ITfDisplayAttributeInfo` implementations.
//!
//! Pattern from karukan-tsf (MIT/Apache-2.0). Two attribute infos:
//!   * Input       — dotted underline (composing romaji / hiragana)
//!   * Converted   — solid bold underline (selected conversion candidate)
//!
//! The GUIDs are registered in `registration.rs` via the
//! `GUID_TFCAT_DISPLAYATTRIBUTEPROVIDER` category. The actual atoms are
//! looked up at Activate time via `ITfCategoryMgr::RegisterGUID`.

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::UI::TextServices::*;

use crate::globals::*;
use crate::tsf::text_input_processor::SetsunaTextService_Impl;

#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl ITfDisplayAttributeProvider_Impl for SetsunaTextService_Impl {
    fn EnumDisplayAttributeInfo(&self) -> Result<IEnumTfDisplayAttributeInfo> {
        let attrs: Vec<ITfDisplayAttributeInfo> =
            vec![InputAttribute.into(), ConvertedAttribute.into()];
        Ok(DisplayAttributeEnum::new(attrs).into())
    }

    fn GetDisplayAttributeInfo(&self, guid: *const GUID) -> Result<ITfDisplayAttributeInfo> {
        unsafe {
            if guid.is_null() {
                return Err(E_INVALIDARG.into());
            }
            let g = *guid;
            if g == GUID_DISPLAY_ATTRIBUTE_INPUT {
                Ok(InputAttribute.into())
            } else if g == GUID_DISPLAY_ATTRIBUTE_CONVERTED {
                Ok(ConvertedAttribute.into())
            } else {
                Err(E_INVALIDARG.into())
            }
        }
    }
}

#[implement(ITfDisplayAttributeInfo)]
struct InputAttribute;

impl ITfDisplayAttributeInfo_Impl for InputAttribute_Impl {
    fn GetGUID(&self) -> Result<GUID> {
        Ok(GUID_DISPLAY_ATTRIBUTE_INPUT)
    }
    fn GetDescription(&self) -> Result<BSTR> {
        Ok(BSTR::from("Setsuna Input"))
    }
    fn GetAttributeInfo(&self, pda: *mut TF_DISPLAYATTRIBUTE) -> Result<()> {
        unsafe {
            if pda.is_null() {
                return Err(E_POINTER.into());
            }
            *pda = TF_DISPLAYATTRIBUTE {
                crText: TF_DA_COLOR {
                    r#type: TF_CT_NONE,
                    ..Default::default()
                },
                crBk: TF_DA_COLOR {
                    r#type: TF_CT_NONE,
                    ..Default::default()
                },
                lsStyle: TF_LS_DOT,
                fBoldLine: FALSE,
                crLine: TF_DA_COLOR {
                    r#type: TF_CT_NONE,
                    ..Default::default()
                },
                bAttr: TF_ATTR_INPUT,
            };
        }
        Ok(())
    }
    fn SetAttributeInfo(&self, _pda: *const TF_DISPLAYATTRIBUTE) -> Result<()> {
        Ok(())
    }
    fn Reset(&self) -> Result<()> {
        Ok(())
    }
}

#[implement(ITfDisplayAttributeInfo)]
struct ConvertedAttribute;

impl ITfDisplayAttributeInfo_Impl for ConvertedAttribute_Impl {
    fn GetGUID(&self) -> Result<GUID> {
        Ok(GUID_DISPLAY_ATTRIBUTE_CONVERTED)
    }
    fn GetDescription(&self) -> Result<BSTR> {
        Ok(BSTR::from("Setsuna Converted"))
    }
    fn GetAttributeInfo(&self, pda: *mut TF_DISPLAYATTRIBUTE) -> Result<()> {
        unsafe {
            if pda.is_null() {
                return Err(E_POINTER.into());
            }
            *pda = TF_DISPLAYATTRIBUTE {
                crText: TF_DA_COLOR {
                    r#type: TF_CT_NONE,
                    ..Default::default()
                },
                crBk: TF_DA_COLOR {
                    r#type: TF_CT_NONE,
                    ..Default::default()
                },
                lsStyle: TF_LS_SOLID,
                fBoldLine: TRUE,
                crLine: TF_DA_COLOR {
                    r#type: TF_CT_NONE,
                    ..Default::default()
                },
                bAttr: TF_ATTR_TARGET_CONVERTED,
            };
        }
        Ok(())
    }
    fn SetAttributeInfo(&self, _pda: *const TF_DISPLAYATTRIBUTE) -> Result<()> {
        Ok(())
    }
    fn Reset(&self) -> Result<()> {
        Ok(())
    }
}

#[implement(IEnumTfDisplayAttributeInfo)]
struct DisplayAttributeEnum {
    attrs: Vec<ITfDisplayAttributeInfo>,
    index: std::cell::Cell<usize>,
}

impl DisplayAttributeEnum {
    fn new(attrs: Vec<ITfDisplayAttributeInfo>) -> Self {
        Self {
            attrs,
            index: std::cell::Cell::new(0),
        }
    }
}

impl IEnumTfDisplayAttributeInfo_Impl for DisplayAttributeEnum_Impl {
    fn Clone(&self) -> Result<IEnumTfDisplayAttributeInfo> {
        let cloned = DisplayAttributeEnum {
            attrs: self.attrs.clone(),
            index: self.index.clone(),
        };
        Ok(cloned.into())
    }

    fn Next(
        &self,
        celt: u32,
        rginfo: *mut Option<ITfDisplayAttributeInfo>,
        pcfetched: *mut u32,
    ) -> Result<()> {
        let mut fetched = 0u32;
        let idx = self.index.get();
        for i in 0..celt as usize {
            let pos = idx + i;
            if pos >= self.attrs.len() {
                break;
            }
            unsafe {
                rginfo.add(i).write(Some(self.attrs[pos].clone()));
            }
            fetched += 1;
        }
        self.index.set(idx + fetched as usize);
        unsafe {
            if !pcfetched.is_null() {
                *pcfetched = fetched;
            }
        }
        if fetched == celt {
            Ok(())
        } else {
            Err(S_FALSE.into())
        }
    }

    fn Reset(&self) -> Result<()> {
        self.index.set(0);
        Ok(())
    }

    fn Skip(&self, ulcount: u32) -> Result<()> {
        let new_idx = self.index.get() + ulcount as usize;
        self.index.set(new_idx.min(self.attrs.len()));
        Ok(())
    }
}
