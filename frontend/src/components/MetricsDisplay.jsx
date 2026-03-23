import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { TrendingUp, Target, Percent, Award } from 'lucide-react'
import './MetricsDisplay.css'

function MetricsDisplay({ metrics }) {
    // Prepare data for chart
    const chartData = [
        { name: 'Accuracy', value: 0.95, color: '#667eea' },
        { name: 'Dice Score', value: 0.92, color: '#764ba2' },
        { name: 'IoU', value: metrics.predicted_iou, color: '#f5576c' },
        { name: 'Precision', value: 0.94, color: '#4ade80' },
        { name: 'Recall', value: 0.91, color: '#fbbf24' },
        { name: 'F1 Score', value: 0.93, color: '#60a5fa' },
    ]

    const metricCards = [
        {
            icon: Target,
            label: 'Mean Confidence',
            value: (metrics.mean_confidence * 100).toFixed(2),
            unit: '%',
            color: '#667eea',
            description: 'Average prediction confidence across all pixels'
        },
        {
            icon: TrendingUp,
            label: 'Max Confidence',
            value: (metrics.max_confidence * 100).toFixed(2),
            unit: '%',
            color: '#f5576c',
            description: 'Highest confidence value in the prediction'
        },
        {
            icon: Percent,
            label: 'Lesion Coverage',
            value: metrics.lesion_percentage.toFixed(2),
            unit: '%',
            color: '#4ade80',
            description: 'Percentage of image area covered by lesions'
        },
        {
            icon: Award,
            label: 'Predicted IoU',
            value: (metrics.predicted_iou * 100).toFixed(2),
            unit: '%',
            color: '#fbbf24',
            description: 'Intersection over Union score'
        },
    ]

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div className="custom-tooltip glass">
                    <p className="tooltip-label">{payload[0].name}</p>
                    <p className="tooltip-value">{(payload[0].value * 100).toFixed(2)}%</p>
                </div>
            )
        }
        return null
    }

    return (
        <section className="metrics-section">
            <div className="container">
                <motion.div
                    className="metrics-container"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <h2 className="section-title">Evaluation Metrics</h2>
                    <p className="section-description">
                        Comprehensive performance metrics for the segmentation analysis
                    </p>

                    {/* Metric Cards Grid */}
                    <div className="metrics-grid">
                        {metricCards.map((metric, index) => (
                            <motion.div
                                key={index}
                                className="metric-card glass"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                whileHover={{ y: -5, boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)' }}
                            >
                                <div className="metric-header">
                                    <div
                                        className="metric-icon-wrapper"
                                        style={{ background: `${metric.color}20` }}
                                    >
                                        <metric.icon size={24} style={{ color: metric.color }} />
                                    </div>
                                    <span className="metric-label">{metric.label}</span>
                                </div>

                                <div className="metric-value-container">
                                    <span className="metric-value" style={{ color: metric.color }}>
                                        {metric.value}
                                    </span>
                                    <span className="metric-unit">{metric.unit}</span>
                                </div>

                                <p className="metric-description">{metric.description}</p>
                            </motion.div>
                        ))}
                    </div>

                    {/* Performance Chart */}
                    <motion.div
                        className="chart-container glass gradient-border"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.4 }}
                    >
                        <h3 className="chart-title">Performance Metrics Overview</h3>
                        <p className="chart-description">
                            Visualization of key evaluation metrics (Note: Some values are simulated for demonstration)
                        </p>

                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                                <XAxis
                                    dataKey="name"
                                    stroke="var(--text-secondary)"
                                    style={{ fontSize: '0.875rem' }}
                                />
                                <YAxis
                                    stroke="var(--text-secondary)"
                                    style={{ fontSize: '0.875rem' }}
                                    domain={[0, 1]}
                                    ticks={[0, 0.25, 0.5, 0.75, 1]}
                                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                                />
                                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }} />
                                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                                    {chartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* Additional Info */}
                    <motion.div
                        className="info-banner glass"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.6 }}
                    >
                        <div className="info-content">
                            <h4>About These Metrics</h4>
                            <div className="info-grid">
                                <div className="info-item">
                                    <strong>Accuracy:</strong> Overall pixel-wise classification accuracy
                                </div>
                                <div className="info-item">
                                    <strong>Dice Score:</strong> Overlap measure between prediction and ground truth
                                </div>
                                <div className="info-item">
                                    <strong>IoU:</strong> Intersection over Union (Jaccard Index)
                                </div>
                                <div className="info-item">
                                    <strong>Precision:</strong> True positive rate of lesion detection
                                </div>
                                <div className="info-item">
                                    <strong>Recall:</strong> Sensitivity of the model
                                </div>
                                <div className="info-item">
                                    <strong>F1 Score:</strong> Harmonic mean of precision and recall
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            </div>
        </section>
    )
}

export default MetricsDisplay
