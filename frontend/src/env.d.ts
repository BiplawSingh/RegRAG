/// <reference types="vite/client" />

// Declares the environment variables this app reads, so a typo such as
// import.meta.env.VITE_API_ULR is a compile error instead of a silent undefined.
interface ImportMetaEnv {
  readonly VITE_API_URL?: string
}
