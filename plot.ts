// Enhanced SVG Plotting Library - Replot
// This code is AI-generated and may not work as expected 
// I'll Add a better version later 

export type Point = {
  x: number;
  y: number;
}

export type PlotType = "line" | "bar" | "scatter";
export type AxisType = "linear" | "logarithmic";

export interface Plot {
  type: PlotType;
  points: Point[];
  color?: string;
  width?: number;
  label?: string;
  fill?: string;
  opacity?: number;
}

export interface AxisConfig {
  type: AxisType;
  label?: string;
  min?: number;
  max?: number;
  showGrid?: boolean;
  gridColor?: string;
}

export class Replot {
  private plots: Plot[] = [];
  private title: string = "";
  private xAxis: AxisConfig = { type: "linear", showGrid: true, gridColor: "#e0e0e0" };
  private yAxis: AxisConfig = { type: "linear", showGrid: true, gridColor: "#e0e0e0" };

  constructor(
    public width: number = 600,
    public height: number = 400,
    public padding: number = 60,
    public backgroundColor: string = "#ffffff",
    public defaultColor: string = "#0074d9"
  ) {}

  addPlot(type: PlotType, points: Point[], options: {
    color?: string;
    width?: number;
    label?: string;
    fill?: string;
    opacity?: number;
  } = {}) {
    if (points.length === 0) {
      console.warn("Cannot add plot with empty points array");
      return -1;
    }

    const plot: Plot = {
      type,
      points,
      color: options.color || this.defaultColor,
      width: options.width || 2,
      label: options.label,
      fill: options.fill || (type === "bar" ? options.color || this.defaultColor : "none"),
      opacity: options.opacity || 1
    };

    return this.plots.push(plot);
  }

  setTitle(title: string) {
    this.title = title;
  }

  setXAxis(config: Partial<AxisConfig>) {
    this.xAxis = { ...this.xAxis, ...config };
  }

  setYAxis(config: Partial<AxisConfig>) {
    this.yAxis = { ...this.yAxis, ...config };
  }

  plot(): string {
    if (this.plots.length === 0) {
      return this.createEmptySVG();
    }

    const { minX, maxX, minY, maxY } = this.calculateBounds();
    const xRange = maxX - minX || 1; // Avoid division by zero
    const yRange = maxY - minY || 1;

    const svgParts: string[] = [
      this.createBackground(),
      this.createGrid(minX, maxX, minY, maxY, xRange, yRange),
      ...this.plots.map(plot => this.renderPlot(plot, minX, maxX, minY, maxY, xRange, yRange)),
      this.createAxes(minX, maxX, minY, maxY, xRange, yRange),
      this.createTitle()
    ];

    return this.createSVGWrapper(svgParts.join('\n'));
  }

  private calculateBounds(): { minX: number; maxX: number; minY: number; maxY: number } {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    this.plots.forEach(plot => {
      plot.points.forEach(point => {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
      });
    });

    // Add some padding to the bounds for better visualization
    const xPadding = (maxX - minX) * 0.05;
    const yPadding = (maxY - minY) * 0.05;

    return {
      minX: minX - xPadding,
      maxX: maxX + xPadding,
      minY: minY - yPadding,
      maxY: maxY + yPadding
    };
  }

  private scaleX(x: number, minX: number, xRange: number): number {
    return this.padding + ((x - minX) / xRange) * (this.width - 2 * this.padding);
  }

  private scaleY(y: number, minY: number, yRange: number): number {
    return this.height - this.padding - ((y - minY) / yRange) * (this.height - 2 * this.padding);
  }

  private createSVGWrapper(content: string): string {
    return `<svg viewBox="0 0 ${this.width} ${this.height}" xmlns="http://www.w3.org/2000/svg">
${content}
</svg>`;
  }

  private createBackground(): string {
    return `<rect width="100%" height="100%" fill="${this.backgroundColor}" />`;
  }

  private createGrid(minX: number, maxX: number, minY: number, maxY: number, xRange: number, yRange: number): string {
    if (!this.xAxis.showGrid && !this.yAxis.showGrid) return '';

    const gridLines: string[] = [];
    
    if (this.yAxis.showGrid) {
      // Vertical grid lines
      for (let x = Math.ceil(minX); x <= Math.floor(maxX); x++) {
        const scaledX = this.scaleX(x, minX, xRange);
        gridLines.push(
          `<line x1="${scaledX}" y1="${this.padding}" x2="${scaledX}" y2="${this.height - this.padding}" stroke="${this.xAxis.gridColor}" stroke-width="1" stroke-dasharray="2,2" />`
        );
      }
    }

    if (this.xAxis.showGrid) {
      // Horizontal grid lines
      for (let y = Math.ceil(minY); y <= Math.floor(maxY); y++) {
        const scaledY = this.scaleY(y, minY, yRange);
        gridLines.push(
          `<line x1="${this.padding}" y1="${scaledY}" x2="${this.width - this.padding}" y2="${scaledY}" stroke="${this.yAxis.gridColor}" stroke-width="1" stroke-dasharray="2,2" />`
        );
      }
    }

    return gridLines.join('\n');
  }

  private createAxes(minX: number, maxX: number, minY: number, maxY: number, xRange: number, yRange: number): string {
    const axes: string[] = [];

    // X-axis
    const xAxisY = this.scaleY(0, minY, yRange);
    axes.push(
      `<line x1="${this.padding}" y1="${xAxisY}" x2="${this.width - this.padding}" y2="${xAxisY}" stroke="#000" stroke-width="2" />`
    );

    // Y-axis
    const yAxisX = this.scaleX(0, minX, xRange);
    axes.push(
      `<line x1="${yAxisX}" y1="${this.padding}" x2="${yAxisX}" y2="${this.height - this.padding}" stroke="#000" stroke-width="2" />`
    );

    // Axis labels
    if (this.xAxis.label) {
      axes.push(
        `<text x="${this.width / 2}" y="${this.height - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">${this.xAxis.label}</text>`
      );
    }

    if (this.yAxis.label) {
      axes.push(
        `<text x="10" y="${this.height / 2}" text-anchor="middle" transform="rotate(-90, 10, ${this.height / 2})" font-family="Arial, sans-serif" font-size="12">${this.yAxis.label}</text>`
      );
    }

    return axes.join('\n');
  }

  private createTitle(): string {
    if (!this.title) return '';
    return `<text x="${this.width / 2}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">${this.title}</text>`;
  }

  private createEmptySVG(): string {
    return this.createSVGWrapper(`
      ${this.createBackground()}
      <text x="${this.width / 2}" y="${this.height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#666">
        No plots to display
      </text>
    `);
  }

  private renderPlot(plot: Plot, minX: number, maxX: number, minY: number, maxY: number, xRange: number, yRange: number): string {
    switch (plot.type) {
      case "line":
        return this.renderLinePlot(plot, minX, maxX, minY, maxY, xRange, yRange);
      case "bar":
        return this.renderBarPlot(plot, minX, maxX, minY, maxY, xRange, yRange);
      case "scatter":
        return this.renderScatterPlot(plot, minX, maxX, minY, maxY, xRange, yRange);
      default:
        return '';
    }
  }

  private renderLinePlot(plot: Plot, minX: number, maxX: number, minY: number, maxY: number, xRange: number, yRange: number): string {
    const points = plot.points.map(p => 
      `${this.scaleX(p.x, minX, xRange)},${this.scaleY(p.y, minY, yRange)}`
    ).join(' ');

    return `<polyline 
      fill="none" 
      stroke="${plot.color}" 
      stroke-width="${plot.width}" 
      stroke-opacity="${plot.opacity}"
      points="${points}"
    />`;
  }

  private renderBarPlot(plot: Plot, minX: number, maxX: number, minY: number, maxY: number, xRange: number, yRange: number): string {
    const barWidth = (this.width - 2 * this.padding) / plot.points.length * 0.8;
    const zeroY = this.scaleY(0, minY, yRange);

    return plot.points.map((point, index) => {
      const x = this.scaleX(point.x, minX, xRange) - barWidth / 2;
      const y = this.scaleY(point.y, minY, yRange);
      const height = Math.abs(zeroY - y);
      const barY = point.y >= 0 ? y : zeroY;

      return `<rect 
        x="${x}" 
        y="${barY}" 
        width="${barWidth}" 
        height="${height}"
        fill="${plot.fill}" 
        fill-opacity="${plot.opacity}"
        stroke="${plot.color}" 
        stroke-width="1"
      />`;
    }).join('\n');
  }

  private renderScatterPlot(plot: Plot, minX: number, maxX: number, minY: number, maxY: number, xRange: number, yRange: number): string {
    return plot.points.map(point => {
      const cx = this.scaleX(point.x, minX, xRange);
      const cy = this.scaleY(point.y, minY, yRange);
      const radius = plot.width || 3;

      return `<circle 
        cx="${cx}" 
        cy="${cy}" 
        r="${radius}" 
        fill="${plot.color}" 
        fill-opacity="${plot.opacity}"
        stroke="${plot.color}" 
        stroke-width="1"
      />`;
    }).join('\n');
  }

  // Utility method to clear all plots
  clear() {
    this.plots = [];
    this.title = "";
  }

  // Method to get current plot count
  getPlotCount(): number {
    return this.plots.length;
  }
}

// Example usage:
/*
const plotter = new Replot(800, 600, 80, "#f8f9fa", "#0074d9");

// Configure axes
plotter.setXAxis({ label: "X Values", showGrid: true });
plotter.setYAxis({ label: "Y Values", showGrid: true });
plotter.setTitle("Sample Plot");

// Add some data
const lineData = [
  { x: 0, y: 0 },
  { x: 1, y: 2 },
  { x: 2, y: 1 },
  { x: 3, y: 4 },
  { x: 4, y: 3 }
];

const barData = [
  { x: 0.5, y: 1 },
  { x: 1.5, y: 3 },
  { x: 2.5, y: 2 },
  { x: 3.5, y: 5 }
];

plotter.addPlot("line", lineData, { color: "#0074d9", width: 3, label: "Line Series" });
plotter.addPlot("bar", barData, { color: "#ff4136", fill: "#ff4136", opacity: 0.7, label: "Bar Series" });

const svg = plotter.plot();
*/