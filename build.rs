#[cfg(target_os = "windows")]
extern crate winres;

fn main() {
    // Warn at build time if the face-detection model is missing.
    // The model is never auto-downloaded — placing it is a one-time manual step.
    let model = std::path::Path::new("assets/seeta_fd_frontal_v1.0.bin");
    if !model.exists() {
        println!("cargo:warning=Face model not found at assets/seeta_fd_frontal_v1.0.bin — eye focus (E key) will be disabled at runtime.");
        println!("cargo:warning=To enable: mkdir assets && curl -sSL -o assets/seeta_fd_frontal_v1.0.bin https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin");
    }

    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("lightningview.ico"); // Replace this with the filename of your .ico file.
        res.compile().unwrap();
    }
}
