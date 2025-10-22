type Point = {
  x: number;
  y: number;
};

type PlotType = "line" | "bar" | "scatter";

interface PlotStyle {
  strokeColor?: string;
  strokeWidth?: number;
  fillColor?: string;
  pointRadius?: number;
}

interface Plot {
  type: PlotType;
  points: Point[];
  style?: PlotStyle;
  label?: string;
}

interface ReplotConfig {
  width?: number;
  height?: number;
  padding?: number;
  strokeColor?: string;
  strokeWidth?: number;
  backgroundColor?: string;
  showGrid?: boolean;
  showAxes?: boolean;
  title?: string;
  xLabel?: string;
  yLabel?: string;
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export class Replot {
  plots: Plot[] = [];
  width: number;
  height: number;
  padding: number;
  strokeColor: string;
  strokeWidth: number;
  backgroundColor: string;
  showGrid: boolean;
  showAxes: boolean;
  title?: string;
  xLabel?: string;
  yLabel?: string;

  constructor(config: ReplotConfig = {}) {
    this.width = config.width ?? 600;
    this.height = config.height ?? 400;
    this.padding = config.padding ?? 50;
    this.strokeColor = config.strokeColor ?? "#0074d9";
    this.strokeWidth = config.strokeWidth ?? 2;
    this.backgroundColor = config.backgroundColor ?? "#ffffff";
    this.showGrid = config.showGrid ?? true;
    this.showAxes = config.showAxes ?? true;
    this.title = config.title;
    this.xLabel = config.xLabel;
    this.yLabel = config.yLabel;
  }

  addPlot(type: PlotType, points: Point[], style?: PlotStyle, label?: string): number {
    if (points.length === 0) {
      throw new Error("Cannot add plot with empty points array");
    }
    return this.plots.push({ type, points, style, label });
  }

  plot(): string {
    if (this.plots.length === 0) {
      return this.emptyPlot();
    }

    const bounds = this.calculateBounds();
    const plotArea = this.getPlotArea();

    const svg = `
      <svg viewBox="0 0 ${this.width} ${this.height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="${this.width}" height="${this.height}" fill="${this.backgroundColor}"/>
        ${this.title ? this.renderTitle() : ""}
        ${this.showGrid ? this.renderGrid(bounds, plotArea) : ""}
        ${this.showAxes ? this.renderAxes(bounds, plotArea) : ""}
        <g clip-path="url(#plot-area)">
          <clipPath id="plot-area">
            <rect x="${plotArea.x}" y="${plotArea.y}" width="${plotArea.width}" height="${plotArea.height}"/>
          </clipPath>
          ${this.plots.map((plt) => this.renderPlot(plt, bounds, plotArea)).join("")}
        </g>
        ${this.xLabel ? this.renderXLabel() : ""}
        ${this.yLabel ? this.renderYLabel() : ""}
        ${this.renderLegend()}
      </svg>
    `;
    return svg;
  }

  private emptyPlot(): string {
    return `
      <svg viewBox="0 0 ${this.width} ${this.height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="${this.width}" height="${this.height}" fill="${this.backgroundColor}"/>
        <text x="${this.width / 2}" y="${this.height / 2}" text-anchor="middle" fill="#666">
          No data to display
        </text>
      </svg>
    `;
  }

  private calculateBounds(): Bounds {
    const allPoints = this.plots.flatMap((p) => p.points);
    
    return {
      minX: Math.min(...allPoints.map((p) => p.x)),
      maxX: Math.max(...allPoints.map((p) => p.x)),
      minY: Math.min(...allPoints.map((p) => p.y)),
      maxY: Math.max(...allPoints.map((p) => p.y)),
    };
  }

  private getPlotArea() {
    const topPadding = this.title ? this.padding + 20 : this.padding;
    return {
      x: this.padding,
      y: topPadding,
      width: this.width - 2 * this.padding,
      height: this.height - topPadding - this.padding,
    };
  }

  private scaleX(x: number, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): number {
    const range = bounds.maxX - bounds.minX || 1;
    return plotArea.x + ((x - bounds.minX) / range) * plotArea.width;
  }

  private scaleY(y: number, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): number {
    const range = bounds.maxY - bounds.minY || 1;
    // Invert Y axis (SVG coordinates are top-down)
    return plotArea.y + plotArea.height - ((y - bounds.minY) / range) * plotArea.height;
  }

  private renderPlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    switch (plot.type) {
      case "line":
        return this.renderLinePlot(plot, bounds, plotArea);
      case "bar":
        return this.renderBarPlot(plot, bounds, plotArea);
      case "scatter":
        return this.renderScatterPlot(plot, bounds, plotArea);
      default:
        return "";
    }
  }

  private renderLinePlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const strokeColor = plot.style?.strokeColor ?? this.strokeColor;
    const strokeWidth = plot.style?.strokeWidth ?? this.strokeWidth;
    
    const points = plot.points
      .map((p) => `${this.scaleX(p.x, bounds, plotArea)},${this.scaleY(p.y, bounds, plotArea)}`)
      .join(" ");

    return `<polyline 
      fill="none" 
      stroke="${strokeColor}" 
      stroke-width="${strokeWidth}" 
      points="${points}"
    />`;
  }

  private renderBarPlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const fillColor = plot.style?.fillColor ?? plot.style?.strokeColor ?? this.strokeColor;
    const strokeColor = plot.style?.strokeColor ?? this.strokeColor;
    const barWidth = plotArea.width / (plot.points.length * 2);

    return plot.points
      .map((p) => {
        const x = this.scaleX(p.x, bounds, plotArea) - barWidth / 2;
        const y = this.scaleY(p.y, bounds, plotArea);
        const baseY = this.scaleY(0, bounds, plotArea);
        const height = Math.abs(baseY - y);

        return `<rect 
          x="${x}" 
          y="${Math.min(y, baseY)}" 
          width="${barWidth}" 
          height="${height}" 
          fill="${fillColor}" 
          stroke="${strokeColor}" 
          stroke-width="1"
        />`;
      })
      .join("");
  }

  private renderScatterPlot(plot: Plot, bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const fillColor = plot.style?.fillColor ?? plot.style?.strokeColor ?? this.strokeColor;
    const radius = plot.style?.pointRadius ?? 3;

    return plot.points
      .map((p) => {
        const cx = this.scaleX(p.x, bounds, plotArea);
        const cy = this.scaleY(p.y, bounds, plotArea);
        return `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${fillColor}"/>`;
      })
      .join("");
  }

  private renderGrid(bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const gridLines: string[] = [];
    const gridColor = "#e0e0e0";
    const numLines = 5;

    // Vertical grid lines
    for (let i = 0; i <= numLines; i++) {
      const x = plotArea.x + (i * plotArea.width) / numLines;
      gridLines.push(
        `<line x1="${x}" y1="${plotArea.y}" x2="${x}" y2="${plotArea.y + plotArea.height}" 
          stroke="${gridColor}" stroke-width="1"/>`
      );
    }

    // Horizontal grid lines
    for (let i = 0; i <= numLines; i++) {
      const y = plotArea.y + (i * plotArea.height) / numLines;
      gridLines.push(
        `<line x1="${plotArea.x}" y1="${y}" x2="${plotArea.x + plotArea.width}" y2="${y}" 
          stroke="${gridColor}" stroke-width="1"/>`
      );
    }

    return gridLines.join("");
  }

  private renderAxes(bounds: Bounds, plotArea: ReturnType<typeof this.getPlotArea>): string {
    const axisColor = "#333";
    const tickSize = 5;
    const numTicks = 5;
    const elements: string[] = [];

    // X-axis
    elements.push(
      `<line x1="${plotArea.x}" y1="${plotArea.y + plotArea.height}" 
        x2="${plotArea.x + plotArea.width}" y2="${plotArea.y + plotArea.height}" 
        stroke="${axisColor}" stroke-width="2"/>`
    );

    // Y-axis
    elements.push(
      `<line x1="${plotArea.x}" y1="${plotArea.y}" 
        x2="${plotArea.x}" y2="${plotArea.y + plotArea.height}" 
        stroke="${axisColor}" stroke-width="2"/>`
    );

    // X-axis ticks and labels
    for (let i = 0; i <= numTicks; i++) {
      const x = plotArea.x + (i * plotArea.width) / numTicks;
      const value = bounds.minX + (i * (bounds.maxX - bounds.minX)) / numTicks;
      
      elements.push(
        `<line x1="${x}" y1="${plotArea.y + plotArea.height}" 
          x2="${x}" y2="${plotArea.y + plotArea.height + tickSize}" 
          stroke="${axisColor}" stroke-width="1"/>`
      );
      
      elements.push(
        `<text x="${x}" y="${plotArea.y + plotArea.height + 20}" 
          text-anchor="middle" font-size="12" fill="${axisColor}">
          ${value.toFixed(1)}
        </text>`
      );
    }

    // Y-axis ticks and labels
    for (let i = 0; i <= numTicks; i++) {
      const y = plotArea.y + plotArea.height - (i * plotArea.height) / numTicks;
      const value = bounds.minY + (i * (bounds.maxY - bounds.minY)) / numTicks;
      
      elements.push(
        `<line x1="${plotArea.x - tickSize}" y1="${y}" 
          x2="${plotArea.x}" y2="${y}" 
          stroke="${axisColor}" stroke-width="1"/>`
      );
      
      elements.push(
        `<text x="${plotArea.x - 10}" y="${y + 4}" 
          text-anchor="end" font-size="12" fill="${axisColor}">
          ${value.toFixed(1)}
        </text>`
      );
    }

    return elements.join("");
  }

  private renderTitle(): string {
    return `<text x="${this.width / 2}" y="25" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">
      ${this.title}
    </text>`;
  }

  private renderXLabel(): string {
    return `<text x="${this.width / 2}" y="${this.height - 10}" text-anchor="middle" font-size="14" fill="#333">
      ${this.xLabel}
    </text>`;
  }

  private renderYLabel(): string {
    return `<text x="15" y="${this.height / 2}" text-anchor="middle" font-size="14" fill="#333" 
      transform="rotate(-90 15 ${this.height / 2})">
      ${this.yLabel}
    </text>`;
  }

  private renderLegend(): string {
    const plotsWithLabels = this.plots.filter((p) => p.label);
    if (plotsWithLabels.length === 0) return "";

    const legendX = this.width - this.padding - 120;
    const legendY = this.padding + (this.title ? 20 : 0);
    const lineHeight = 20;

    const items = plotsWithLabels.map((plot, i) => {
      const y = legendY + i * lineHeight;
      const color = plot.style?.strokeColor ?? plot.style?.fillColor ?? this.strokeColor;
      
      return `
        <rect x="${legendX}" y="${y}" width="15" height="3" fill="${color}"/>
        <text x="${legendX + 20}" y="${y + 4}" font-size="12" fill="#333">${plot.label}</text>
      `;
    });

    return `<g class="legend">${items.join("")}</g>`;
  }

  // Utility method to clear all plots
  clear(): void {
    this.plots = [];
  }
}