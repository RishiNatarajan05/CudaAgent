import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CUDA Agent",
  description: "CUDA-aware code intelligence powered by Claude",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
