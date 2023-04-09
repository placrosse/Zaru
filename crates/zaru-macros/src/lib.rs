//! Procedural macros used by `zaru`.
//!
//! Do not use this crate directly, use `zaru` instead.

use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::{parse::Error, ItemFn};

#[proc_macro_attribute]
pub fn main(args: TokenStream, item: TokenStream) -> TokenStream {
    match expand_main(args, item.clone()) {
        Ok(tokens) => tokens,
        Err(err) => {
            // Emit the `compile_error!` invocation, alongside the original item, in an attempt to
            // improve IDE support.
            let mut error = item.clone();
            error.extend(TokenStream::from(err.to_compile_error()));
            error
        }
    }
}

fn expand_main(args: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    if !args.is_empty() {
        return Err(Error::new(
            Span::call_site().into(),
            "`#[zaru::main]` does not accept arguments",
        ));
    }

    let item = syn::parse::<ItemFn>(item)?;

    if item.sig.ident != "main" {
        return Err(Error::new(
            item.sig.ident.span(),
            "`#[zaru::main]` must be applied to a function called `main`",
        ));
    }

    Ok(quote! {
        fn main() {
            #item

            ::zaru::init_logger!();

            zaru::run(main);
        }
    }
    .into())
}
