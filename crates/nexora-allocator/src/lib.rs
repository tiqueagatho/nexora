//! Asignadores de memoria y arena para KV-cache.
//!
//! Este crate provee:
//! - Integración con mimalloc/jemalloc como allocador global
//! - Un arena allocator de bump para KV-cache (O(1) alloc/free)
//!
//! # Features
//!
//! - `mimalloc`: Usa mimalloc como allocador global (recomendado)
//! - `jemalloc`: Usa jemalloc como allocador global

// ── Global Allocator ──────────────────────────────────────

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ── KV-Cache Arena ────────────────────────────────────────

/// Arena allocator de bump para KV-cache.
///
/// Asigna memoria de un bloque pre-allocado con O(1) alloc y O(1) free
/// (reset completo). Ideal para buffers de inferencia donde se asigna
/// mucho al inicio y se libera todo de golpe al final.
///
/// # Ejemplo
///
/// ```ignore
/// let mut arena = KVCacheArena::new(256); // 256 MB
/// let slice = arena.alloc_f32_slice(4096); // 4096 floats
/// // ... usar slice ...
/// arena.reset(); // Liberar toda la memoria de golpe
/// ```
pub struct KVCacheArena {
    buffer: Vec<u8>,
    offset: usize,
}

impl KVCacheArena {
    /// Crear una arena con capacidad en megabytes.
    pub fn new(capacity_mb: usize) -> Self {
        let capacity = capacity_mb * 1024 * 1024;
        Self {
            buffer: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Crear una arena con capacidad en bytes.
    pub fn with_bytes(capacity_bytes: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity_bytes],
            offset: 0,
        }
    }

    /// Asignar un slice de `f32` con alineación correcta.
    #[inline(always)]
    pub fn alloc_f32_slice(&mut self, len: usize) -> &mut [f32] {
        let size = len * std::mem::size_of::<f32>();
        let align = std::mem::align_of::<f32>();
        let aligned_offset = (self.offset + align - 1) & !(align - 1);
        let start = aligned_offset;
        self.offset = aligned_offset + size;

        assert!(
            self.offset <= self.buffer.len(),
            "KVCacheArena out of memory: requested {} bytes, capacity {} bytes",
            size,
            self.buffer.len()
        );

        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr().add(start) as *mut f32, len)
        }
    }

    /// Asignar un slice de `u8`.
    #[inline(always)]
    pub fn alloc_u8_slice(&mut self, len: usize) -> &mut [u8] {
        let start = self.offset;
        self.offset += len;

        assert!(
            self.offset <= self.buffer.len(),
            "KVCacheArena out of memory"
        );

        &mut self.buffer[start..self.offset]
    }

    /// Reset completo: libera toda la memoria asignada.
    /// O(1) — solo mueve el puntero al inicio.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Bytes usados actualmente.
    #[inline]
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Bytes totales disponibles.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Porcentaje de uso.
    #[inline]
    pub fn usage_ratio(&self) -> f64 {
        self.offset as f64 / self.buffer.len() as f64
    }
}

unsafe impl Send for KVCacheArena {}
unsafe impl Sync for KVCacheArena {}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_arena_basic() {
        let mut arena = KVCacheArena::new(1); // 1 MB
        assert_eq!(arena.used(), 0);

        let slice = arena.alloc_f32_slice(1024);
        assert_eq!(slice.len(), 1024);
        assert!(arena.used() > 0);

        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn kv_cache_arena_alignment() {
        let mut arena = KVCacheArena::new(1);
        let slice = arena.alloc_f32_slice(256);

        // Verify alignment (pointer should be aligned to f32)
        let ptr = slice.as_ptr() as usize;
        assert_eq!(ptr % std::mem::align_of::<f32>(), 0);
    }

    #[test]
    #[should_panic(expected = "KVCacheArena out of memory")]
    fn kv_cache_arena_overflow() {
        let mut arena = KVCacheArena::with_bytes(16);
        // Try to allocate more than capacity
        arena.alloc_f32_slice(1024); // 4096 bytes > 16 bytes
    }
}
