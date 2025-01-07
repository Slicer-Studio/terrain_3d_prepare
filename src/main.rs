use eframe::{run_native, App, Frame, NativeOptions};
use egui::{CentralPanel, Context, ComboBox, ColorImage, TextureHandle, Vec2, widgets::Image, load::SizedTexture, CollapsingHeader};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use image::{DynamicImage, ImageBuffer, GenericImageView};
use rayon::prelude::*;
use image_dds::{dds_from_image, Quality, Mipmaps};
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, PartialEq, Clone, Copy)]
enum NormalMapFormat {
    OpenGL,
    DirectX,
}

impl Default for NormalMapFormat {
    fn default() -> Self {
        NormalMapFormat::OpenGL
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum OutputFormat {
    PNG,
    DDS,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::PNG
    }
}

// Add new enum for roughness format
#[derive(Debug, PartialEq, Clone, Copy)]
enum RoughnessFormat {
    Roughness,
    Smoothness,
}

impl Default for RoughnessFormat {
    fn default() -> Self {
        RoughnessFormat::Roughness
    }
}

#[derive(Debug)]
enum ImageLoadState {
    NotLoaded,
    Loading,
    Loaded,
    Error(String),
}

#[derive(Debug, Clone)]
struct ProcessedImage {
    original: DynamicImage,
    downscaled: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

#[derive(Debug)]
enum ImageValidationError {
    NotSquare,
    NotPowerOfTwo,
    TooSmall,
}

impl std::fmt::Display for ImageValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Image must be square"),
            Self::NotPowerOfTwo => write!(f, "Image dimensions must be power of 2"),
            Self::TooSmall => write!(f, "Image must be at least 512x512"),
        }
    }
}

#[derive(Debug)]
enum ProcessingState {
    NotStarted,
    Processing,
    Done,
    Error(String),
}

struct TerrainApp {
    albedo_map: Option<PathBuf>,
    height_map: Option<PathBuf>,
    ambient_occlusion_map: Option<PathBuf>,
    normal_map: Option<PathBuf>,
    normal_map_format: NormalMapFormat,
    albedo_load_state: ImageLoadState,
    height_load_state: ImageLoadState,
    normal_load_state: ImageLoadState,
    ao_load_state: ImageLoadState,
    image_receiver: Receiver<(String, Result<ProcessedImage, String>)>,
    image_sender: Sender<(String, Result<ProcessedImage, String>)>,
    albedo_image: Option<ProcessedImage>,
    height_image: Option<ProcessedImage>,
    normal_image: Option<ProcessedImage>,
    ao_image: Option<ProcessedImage>,
    albedo_texture: Option<TextureHandle>,
    height_texture: Option<TextureHandle>,
    normal_texture: Option<TextureHandle>,
    ao_texture: Option<TextureHandle>,
    output_directory: Option<PathBuf>,
    output_format: OutputFormat,
    processing_state: ProcessingState,
    processing_receiver: Receiver<Result<(), String>>,
    processing_sender: Sender<Result<(), String>>,
    roughness_map: Option<PathBuf>,
    roughness_load_state: ImageLoadState,
    roughness_image: Option<ProcessedImage>,
    roughness_texture: Option<TextureHandle>,
    roughness_format: RoughnessFormat,
}

impl Default for TerrainApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (ptx, prx) = channel();
        Self {
            albedo_map: None,
            height_map: None,
            ambient_occlusion_map: None,
            normal_map: None,
            normal_map_format: Default::default(),
            albedo_load_state: ImageLoadState::NotLoaded,
            height_load_state: ImageLoadState::NotLoaded,
            normal_load_state: ImageLoadState::NotLoaded,
            ao_load_state: ImageLoadState::NotLoaded,
            image_receiver: rx,
            image_sender: tx,
            albedo_image: None,
            height_image: None,
            normal_image: None,
            ao_image: None,
            albedo_texture: None,
            height_texture: None,
            normal_texture: None,
            ao_texture: None,
            output_directory: None,
            output_format: Default::default(),
            processing_state: ProcessingState::NotStarted,
            processing_receiver: prx,
            processing_sender: ptx,
            roughness_map: None,
            roughness_load_state: ImageLoadState::NotLoaded,
            roughness_image: None,
            roughness_texture: None,
            roughness_format: Default::default(),
        }
    }
}

impl TerrainApp {
    const SUPPORTED_FORMATS: [&'static str; 16] = [
        "avif", "bmp", "dds", "exr", "gif", "hdr", "ico", "jpg", "jpeg", 
        "png", "pnm", "qoi", "tga", "tiff", "tif", "webp"
    ];

    fn validate_image(img: &DynamicImage) -> Result<(), ImageValidationError> {
        let (width, height) = img.dimensions();
        
        if width != height {
            return Err(ImageValidationError::NotSquare);
        }
        
        if !width.is_power_of_two() {
            return Err(ImageValidationError::NotPowerOfTwo);
        }
        
        if width < 512 {
            return Err(ImageValidationError::TooSmall);
        }
        
        Ok(())
    }

    fn process_image(img: DynamicImage) -> Result<ProcessedImage, String> {
        Self::validate_image(&img).map_err(|e| e.to_string())?;
        
        let downscaled = img.resize_exact(512, 512, image::imageops::FilterType::Nearest)
            .to_rgba8();
            
        Ok(ProcessedImage {
            original: img,
            downscaled,
        })
    }

    fn load_image(&self, path: PathBuf, image_type: String) {
        let tx = self.image_sender.clone();
        thread::spawn(move || {
            let result = image::open(&path)
                .map_err(|e| e.to_string())
                .and_then(TerrainApp::process_image);
            tx.send((image_type, result)).ok();
        });
    }

    fn process_image_to_texture(&mut self, processed: &ProcessedImage, ctx: &Context) -> TextureHandle {
        let size = [processed.downscaled.width() as _, processed.downscaled.height() as _];
        let pixels = processed.downscaled.as_flat_samples();
        let color_image = ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
        ctx.load_texture("image", color_image, Default::default())
    }

    fn display_image(&self, ui: &mut egui::Ui, texture: &TextureHandle) {
        let available_width = ui.available_width();
        let size = texture.size_vec2();
        let aspect_ratio = size.x / size.y;
        let display_size = Vec2::new(
            available_width,
            available_width / aspect_ratio
        );
        
        ui.add(Image::from_texture(SizedTexture::from_handle(texture)).max_size(display_size));
    }

    fn are_required_images_loaded(&self) -> bool {
        matches!(
            (&self.albedo_load_state, &self.normal_load_state),
            (
                ImageLoadState::Loaded,
                ImageLoadState::Loaded
            )
        ) && self.output_directory.is_some()
    }

    fn save_as_dds(img: &DynamicImage, path: PathBuf) -> Result<(), String> {
        let rgba = img.to_rgba8();
        let dds = dds_from_image(
            &rgba,
            image_dds::ImageFormat::BC3RgbaUnorm,
            Quality::Normal,
            Mipmaps::GeneratedAutomatic,
        ).map_err(|e| format!("Failed to convert to DDS: {}", e))?;

        let file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);
        
        dds.write(&mut writer)
            .map_err(|e| format!("Failed to write DDS: {}", e))
    }

    fn process_and_save_images(&mut self) -> Result<(), String> {
        let output_dir = self.output_directory.as_ref().unwrap().clone();
        let albedo = self.albedo_image.as_ref().unwrap().original.clone();
        let height = self.height_image.as_ref().map(|img| img.original.clone());
        let normal = self.normal_image.as_ref().unwrap().original.clone();
        let ao = self.ao_image.clone();
        let roughness = self.roughness_image.clone();
        let roughness_format = self.roughness_format;
        let normal_format = self.normal_map_format;
        let output_format = self.output_format;
        let tx = self.processing_sender.clone();

        self.processing_state = ProcessingState::Processing;
        
        thread::spawn(move || {
            let result = (move || {
                // Process albedo + AO
                let mut final_texture = albedo.to_rgba8();
                let width = final_texture.width();
                
                // Convert to vec for parallel processing
                let mut pixels: Vec<_> = final_texture.pixels_mut().collect();
                
                // If AO map exists, multiply it with albedo
                if let Some(ao_image) = ao {
                    let ao = ao_image.original.to_luma8();
                    pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
                        let x = (i % width as usize) as u32;
                        let y = (i / width as usize) as u32;
                        let ao_val = ao.get_pixel(x, y)[0] as f32 / 255.0;
                        pixel[0] = (pixel[0] as f32 * ao_val) as u8;
                        pixel[1] = (pixel[1] as f32 * ao_val) as u8;
                        pixel[2] = (pixel[2] as f32 * ao_val) as u8;
                    });
                }

                // Add height as alpha channel if it exists
                if let Some(height_img) = height {
                    let height = height_img.to_luma8();
                    pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
                        let x = (i % width as usize) as u32;
                        let y = (i / width as usize) as u32;
                        pixel[3] = height.get_pixel(x, y)[0];
                    });
                } else {
                    // Set alpha to full opacity if no height map
                    pixels.par_iter_mut().for_each(|pixel| {
                        pixel[3] = 255;
                    });
                }

                // Process normal map with roughness
                let mut normal_image = normal.to_rgba8();
                let width = normal_image.width();
                let height = normal_image.height();  // Get height before mutable borrow
                let mut pixels: Vec<_> = normal_image.pixels_mut().collect();

                // Process DirectX normal map if needed
                if normal_format == NormalMapFormat::DirectX {
                    pixels.par_iter_mut().for_each(|p| {
                        p[1] = 255 - p[1]; // Invert green channel
                    });
                }

                // Add roughness as alpha channel
                if let Some(roughness_img) = roughness {
                    let roughness = roughness_img.original.to_luma8();
                    pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
                        let x = (i % width as usize) as u32;
                        let y = (i / width as usize) as u32;
                        let value = roughness.get_pixel(x, y)[0];
                        // Store as smoothness - invert if it's a roughness map
                        pixel[3] = match roughness_format {
                            RoughnessFormat::Roughness => value, // Invert roughness to smoothness
                            RoughnessFormat::Smoothness => 255 - value, // Keep smoothness as-is
                        };
                    });
                } else {
                    // Set default smoothness if no map provided (0.5)
                    pixels.par_iter_mut().for_each(|pixel| {
                        pixel[3] = 128;
                    });
                }

                // Save images based on format
                match output_format {
                    OutputFormat::PNG => {
                        final_texture.save(output_dir.join("albedo.png"))
                            .map_err(|e| e.to_string())?;
                        
                        // Create RGBA image buffer with explicit type
                        let normal_buffer = ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_vec(
                            width,
                            height,  // Use stored height value
                            pixels.into_iter().flat_map(|p| p.0.to_vec()).collect()
                        ).unwrap();
                        
                        normal_buffer.save(output_dir.join("normal.png"))
                            .map_err(|e| e.to_string())?;
                    }
                    OutputFormat::DDS => {
                        Self::save_as_dds(&final_texture.into(), output_dir.join("albedo.dds"))?;
                        
                        // Create RGBA image buffer with explicit type
                        let normal_buffer = ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_vec(
                            width,
                            height,  // Use stored height value
                            pixels.into_iter().flat_map(|p| p.0.to_vec()).collect()
                        ).unwrap();
                        
                        Self::save_as_dds(
                            &DynamicImage::ImageRgba8(normal_buffer),
                            output_dir.join("normal.dds")
                        )?;
                    }
                }

                Ok(())
            })();

            tx.send(result).ok();
        });

        Ok(())
    }

    // Add new methods to clear image states
    fn clear_height_map(&mut self) {
        self.height_map = None;
        self.height_image = None;
        self.height_texture = None;
        self.height_load_state = ImageLoadState::NotLoaded;
    }

    fn clear_ao_map(&mut self) {
        self.ambient_occlusion_map = None;
        self.ao_image = None;
        self.ao_texture = None;
        self.ao_load_state = ImageLoadState::NotLoaded;
    }

    fn clear_roughness_map(&mut self) {
        self.roughness_map = None;
        self.roughness_image = None;
        self.roughness_texture = None;
        self.roughness_load_state = ImageLoadState::NotLoaded;
    }
}

impl App for TerrainApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // Handle image loading results
        while let Ok((image_type, result)) = self.image_receiver.try_recv() {
            match (image_type.as_str(), result) {
                ("albedo", Ok(processed)) => {
                    self.albedo_texture = Some(self.process_image_to_texture(&processed, ctx));
                    self.albedo_image = Some(processed);
                    self.albedo_load_state = ImageLoadState::Loaded;
                }
                ("height", Ok(processed)) => {
                    self.height_texture = Some(self.process_image_to_texture(&processed, ctx));
                    self.height_image = Some(processed);
                    self.height_load_state = ImageLoadState::Loaded;
                }
                ("normal", Ok(processed)) => {
                    self.normal_texture = Some(self.process_image_to_texture(&processed, ctx));
                    self.normal_image = Some(processed);
                    self.normal_load_state = ImageLoadState::Loaded;
                }
                ("ao", Ok(processed)) => {
                    self.ao_texture = Some(self.process_image_to_texture(&processed, ctx));
                    self.ao_image = Some(processed);
                    self.ao_load_state = ImageLoadState::Loaded;
                }
                ("roughness", Ok(processed)) => {
                    self.roughness_texture = Some(self.process_image_to_texture(&processed, ctx));
                    self.roughness_image = Some(processed);
                    self.roughness_load_state = ImageLoadState::Loaded;
                }
                (type_name, Err(e)) => {
                    match type_name {
                        "albedo" => self.albedo_load_state = ImageLoadState::Error(e),
                        "height" => self.height_load_state = ImageLoadState::Error(e),
                        "normal" => self.normal_load_state = ImageLoadState::Error(e),
                        "ao" => self.ao_load_state = ImageLoadState::Error(e),
                        "roughness" => self.roughness_load_state = ImageLoadState::Error(e),
                        _ => {}
                    }
                }
                _ => {}
            }
            ctx.request_repaint();
        }

        // Handle processing results
        if let Ok(result) = self.processing_receiver.try_recv() {
            self.processing_state = match result {
                Ok(()) => ProcessingState::Done,
                Err(e) => ProcessingState::Error(e),
            };
            ctx.request_repaint();
        }

        CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Terrain 3D Prepare");
                    
                    // Input Section
                    CollapsingHeader::new("Input")
                        .default_open(true)
                        .show(ui, |ui| {
                            // Texture Maps
                            CollapsingHeader::new("Texture Maps")
                                .default_open(true)
                                .show(ui, |ui| {
                                    // Albedo Map
                                    CollapsingHeader::new("Albedo Map (Required)")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            if ui.button("Select Albedo Map").clicked() {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .add_filter("Image files", &Self::SUPPORTED_FORMATS)
                                                    .pick_file() {
                                                    self.albedo_map = Some(path.clone());
                                                    self.albedo_load_state = ImageLoadState::Loading;
                                                    self.load_image(path, "albedo".to_string());
                                                }
                                            }
                                            if let Some(path) = &self.albedo_map {
                                                ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
                                                match &self.albedo_load_state {
                                                    ImageLoadState::Loading => ui.spinner(),
                                                    ImageLoadState::Error(e) => ui.label(format!("Error: {}", e)),
                                                    _ => ui.label(""),
                                                };
                                            }
                                            if let Some(texture) = &self.albedo_texture {
                                                self.display_image(ui, texture);
                                            }
                                        });

                                    // Ambient Occlusion Map
                                    CollapsingHeader::new("AO Map (Optional)")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                if ui.button("Select AO Map").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new()
                                                        .add_filter("Image files", &Self::SUPPORTED_FORMATS)
                                                        .pick_file() {
                                                        self.ambient_occlusion_map = Some(path.clone());
                                                        self.ao_load_state = ImageLoadState::Loading;
                                                        self.load_image(path, "ao".to_string());
                                                    }
                                                }
                                                if ui.button("Clear").clicked() {
                                                    self.clear_ao_map();
                                                }
                                            });
                                            if let Some(path) = &self.ambient_occlusion_map {
                                                ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
                                                match &self.ao_load_state {
                                                    ImageLoadState::Loading => ui.spinner(),
                                                    ImageLoadState::Error(e) => ui.label(format!("Error: {}", e)),
                                                    _ => ui.label(""),
                                                };
                                            }
                                            if let Some(texture) = &self.ao_texture {
                                                self.display_image(ui, texture);
                                            }
                                        });

                                    // Height Map (renamed from Displacement)
                                    CollapsingHeader::new("Height Map (Optional)")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                if ui.button("Select Height Map").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new()
                                                        .add_filter("Image files", &Self::SUPPORTED_FORMATS)
                                                        .pick_file() {
                                                        self.height_map = Some(path.clone());
                                                        self.height_load_state = ImageLoadState::Loading;
                                                        self.load_image(path, "height".to_string());
                                                    }
                                                }
                                                if ui.button("Clear").clicked() {
                                                    self.clear_height_map();
                                                }
                                            });
                                            if let Some(path) = &self.height_map {
                                                ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
                                                match &self.height_load_state {
                                                    ImageLoadState::Loading => ui.spinner(),
                                                    ImageLoadState::Error(e) => ui.label(format!("Error: {}", e)),
                                                    _ => ui.label(""),
                                                };
                                            }
                                            if let Some(texture) = &self.height_texture {
                                                self.display_image(ui, texture);
                                            }
                                        });
                                });

                            // Normal Maps
                            CollapsingHeader::new("Normal Maps")
                                .default_open(true)
                                .show(ui, |ui| {
                                    // Normal Map
                                    CollapsingHeader::new("Normal Map (Required)")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            if ui.button("Select Normal Map").clicked() {
                                                if let Some(path) = rfd::FileDialog::new()
                                                    .add_filter("Image files", &Self::SUPPORTED_FORMATS)
                                                    .pick_file() {
                                                    self.normal_map = Some(path.clone());
                                                    self.normal_load_state = ImageLoadState::Loading;
                                                    self.load_image(path, "normal".to_string());
                                                }
                                            }
                                            ComboBox::from_label("")
                                                .selected_text(format!("{:?}", self.normal_map_format))
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(&mut self.normal_map_format, NormalMapFormat::OpenGL, "OpenGL");
                                                    ui.selectable_value(&mut self.normal_map_format, NormalMapFormat::DirectX, "DirectX");
                                                });
                                            if let Some(path) = &self.normal_map {
                                                ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
                                                match &self.normal_load_state {
                                                    ImageLoadState::Loading => ui.spinner(),
                                                    ImageLoadState::Error(e) => ui.label(format!("Error: {}", e)),
                                                    _ => ui.label(""),
                                                };
                                            }
                                            if let Some(texture) = &self.normal_texture {
                                                self.display_image(ui, texture);
                                            }
                                        });

                                    // Update Roughness Map section to include format dropdown
                                    CollapsingHeader::new("Roughness Map (Optional)")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                if ui.button("Select Roughness Map").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new()
                                                        .add_filter("Image files", &Self::SUPPORTED_FORMATS)
                                                        .pick_file() {
                                                        self.roughness_map = Some(path.clone());
                                                        self.roughness_load_state = ImageLoadState::Loading;
                                                        self.load_image(path, "roughness".to_string());
                                                    }
                                                }
                                                if ui.button("Clear").clicked() {
                                                    self.clear_roughness_map();
                                                }
                                            });
                                            ComboBox::from_label("")
                                                .selected_text(format!("{:?}", self.roughness_format))
                                                .show_ui(ui, |ui| {
                                                    ui.selectable_value(&mut self.roughness_format, RoughnessFormat::Roughness, "Roughness");
                                                    ui.selectable_value(&mut self.roughness_format, RoughnessFormat::Smoothness, "Smoothness");
                                                });
                                            if let Some(path) = &self.roughness_map {
                                                ui.label(path.file_name().unwrap_or_default().to_string_lossy().to_string());
                                                match &self.roughness_load_state {
                                                    ImageLoadState::Loading => ui.spinner(),
                                                    ImageLoadState::Error(e) => ui.label(format!("Error: {}", e)),
                                                    _ => ui.label(""),
                                                };
                                            }
                                            if let Some(texture) = &self.roughness_texture {
                                                self.display_image(ui, texture);
                                            }
                                        });
                                });
                        });

                    // Output Section
                    CollapsingHeader::new("Output")
                        .default_open(true)
                        .show(ui, |ui| {
                            if ui.button("Select Output Directory").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .pick_folder() {
                                    self.output_directory = Some(path);
                                }
                            }
                            if let Some(path) = &self.output_directory {
                                ui.label(path.to_string_lossy().to_string());
                            }
                            
                            ComboBox::from_label("Output Format")
                                .selected_text(format!("{:?}", self.output_format))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.output_format, OutputFormat::PNG, "PNG");
                                    ui.selectable_value(&mut self.output_format, OutputFormat::DDS, "DDS");
                                });
                        });

                    // Show processing status
                    match &self.processing_state {
                        ProcessingState::Processing => {
                            ui.spinner();
                            ui.label("Processing...");
                        }
                        ProcessingState::Done => {
                            ui.label("Processing complete");
                        }
                        ProcessingState::Error(e) => {
                            ui.label(format!("Error: {}", e));
                        }
                        _ => {}
                    }

                    ui.add_space(8.0);
                    let run_button = ui.add_enabled_ui(
                        self.are_required_images_loaded() && 
                        !matches!(self.processing_state, ProcessingState::Processing),
                        |ui| {
                            ui.button("Run")
                        }
                    ).inner;
                    
                    if run_button.clicked() {
                        if let Err(e) = self.process_and_save_images() {
                            self.processing_state = ProcessingState::Error(e);
                        }
                    }
                });
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 800.0]), // Adjusted for vertical layout
        ..Default::default()
    };

    run_native(
        "Terrain 3D Prepare",
        options,
        Box::new(|_cc| Ok(Box::new(TerrainApp::default()))),
    )
}
