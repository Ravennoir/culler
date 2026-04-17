#[cfg(target_os = "windows")]
extern crate winres;

fn main() {
    // If FontAwesome Free Solid is present in assets/, embed it and enable FA icons.
    // Download once with:
    //   mkdir -p assets
    //   curl -Lo assets/fa-solid-900.ttf \
    //     https://github.com/FortAwesome/Font-Awesome/raw/6.x/webfonts/fa-solid-900.ttf
    if std::path::Path::new("assets/fa-solid-900.ttf").exists() {
        println!("cargo:rustc-cfg=has_fontawesome");
    } else {
        println!("cargo:warning=FontAwesome Solid not found — using Unicode fallback glyphs.");
        println!("cargo:warning=Run: mkdir -p assets && curl -Lo assets/fa-solid-900.ttf https://github.com/FortAwesome/Font-Awesome/raw/6.x/webfonts/fa-solid-900.ttf");
    }
    println!("cargo:rerun-if-changed=assets/fa-solid-900.ttf");

    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("lightningview.ico");
        res.compile().unwrap();
    }
}
