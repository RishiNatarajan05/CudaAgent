"use client";
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface RooflineData {
  estimated_ai: number;
  peak_compute_tflops: number;
  peak_bandwidth_tbps: number;
  ridge_point: number;
  is_compute_bound: boolean;
}

export function RooflinePlot({ data }: { data: RooflineData }) {
  const xs = [0.01, 0.1, 1, 10, 100, 1000, 10000];
  const peakCompute = data.peak_compute_tflops; // TFLOP/s
  const bw = data.peak_bandwidth_tbps * 1000; // GB/s
  // Performance(ai) = min(peak_compute, ai * bw_in_GFLOPs_per_s)/ 1000 = TFLOP/s
  const series = xs.map((ai) => {
    const memBound = (ai * bw) / 1000; // TFLOP/s = ai * bytes/s, ai is FLOPs/byte, bw is GB/s = 1e9 bytes/s, /1e12 -> *1e-3
    return {
      ai,
      memRoof: Math.min(memBound, peakCompute),
      computeRoof: peakCompute,
    };
  });

  const kernelPoint = [
    {
      ai: data.estimated_ai,
      perf:
        Math.min((data.estimated_ai * bw) / 1000, peakCompute),
    },
  ];

  return (
    <div className="border border-zinc-800 rounded p-3 bg-zinc-900">
      <div className="text-sm text-zinc-300 mb-2 flex items-center justify-between">
        <span>Roofline (A100 reference)</span>
        <span
          className={`text-xs px-2 py-0.5 rounded ${
            data.is_compute_bound
              ? "bg-amber-900 text-amber-200"
              : "bg-cyan-900 text-cyan-200"
          }`}
        >
          {data.is_compute_bound ? "Compute Bound" : "Memory Bound"}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={series} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="#27272a" />
          <XAxis
            dataKey="ai"
            type="number"
            scale="log"
            domain={[0.01, 10000]}
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            label={{ value: "Arithmetic Intensity (FLOPs/byte)", position: "insideBottom", offset: -2, fill: "#a1a1aa", fontSize: 11 }}
          />
          <YAxis
            type="number"
            scale="log"
            domain={[0.1, peakCompute * 1.5]}
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            label={{ value: "TFLOP/s", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 11 }}
          />
          <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #27272a" }} />
          <Legend wrapperStyle={{ color: "#a1a1aa", fontSize: 11 }} />
          <Line dataKey="memRoof" name="Roofline" stroke="#76b900" dot={false} strokeWidth={2} />
          <Line dataKey="computeRoof" name="Peak compute" stroke="#52525b" dot={false} strokeDasharray="4 4" />
          <Scatter data={kernelPoint} name="Kernel" fill="#f43f5e" line={false} dataKey="perf" />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="text-[11px] text-zinc-500 mt-2">
        Estimated AI: {data.estimated_ai} FLOPs/byte · Ridge point: {data.ridge_point} ·{" "}
        Static analysis estimate. Use Nsight Compute for profiling.
      </div>
    </div>
  );
}
