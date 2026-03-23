import { motion } from 'framer-motion'
import { X, Download, ZoomIn } from 'lucide-react'
import { useState } from 'react'
import './ResultsSection.css'

function ResultsSection({ results, onReset }) {
    const [selectedView, setSelectedView] = useState('visualization')
    const [zoomedImage, setZoomedImage] = useState(null)

    const views = [
        { id: 'original', label: 'Original', image: results.original_image },
        { id: 'mask', label: 'Segmentation Mask', image: results.segmentation_mask },
        { id: 'visualization', label: 'Visualization', image: results.visualization },
        { id: 'confidence', label: 'Confidence Map', image: results.confidence_map },
    ]

    const handleDownload = (imageData, filename) => {
        const link = document.createElement('a')
        link.href = `data:image/png;base64,${imageData}`
        link.download = filename
        link.click()
    }

    return (
        <section className="results-section">
            <div className="container">
                <motion.div
                    className="results-container"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <div className="results-header flex items-center justify-between">
                        <h2 className="section-title">Segmentation Results</h2>
                        <motion.button
                            className="btn btn-secondary"
                            onClick={onReset}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            <X size={20} />
                            <span>New Analysis</span>
                        </motion.button>
                    </div>

                    {/* View Selector */}
                    <div className="view-selector">
                        {views.map((view) => (
                            <motion.button
                                key={view.id}
                                className={`view-btn ${selectedView === view.id ? 'active' : ''}`}
                                onClick={() => setSelectedView(view.id)}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                {view.label}
                            </motion.button>
                        ))}
                    </div>

                    {/* Image Display */}
                    <div className="results-grid">
                        <motion.div
                            className="image-display-card glass gradient-border"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="image-container">
                                <img
                                    src={`data:image/png;base64,${views.find(v => v.id === selectedView)?.image}`}
                                    alt={views.find(v => v.id === selectedView)?.label}
                                    className="result-image"
                                />

                                <div className="image-actions">
                                    <motion.button
                                        className="action-btn"
                                        onClick={() => setZoomedImage(views.find(v => v.id === selectedView))}
                                        whileHover={{ scale: 1.1 }}
                                        whileTap={{ scale: 0.9 }}
                                        title="Zoom"
                                    >
                                        <ZoomIn size={20} />
                                    </motion.button>
                                    <motion.button
                                        className="action-btn"
                                        onClick={() => handleDownload(
                                            views.find(v => v.id === selectedView)?.image,
                                            `${selectedView}.png`
                                        )}
                                        whileHover={{ scale: 1.1 }}
                                        whileTap={{ scale: 0.9 }}
                                        title="Download"
                                    >
                                        <Download size={20} />
                                    </motion.button>
                                </div>
                            </div>

                            <div className="image-info">
                                <h3>{views.find(v => v.id === selectedView)?.label}</h3>
                                <p className="image-description">
                                    {selectedView === 'original' && 'Original uploaded eye image'}
                                    {selectedView === 'mask' && 'Binary segmentation mask highlighting detected lesions'}
                                    {selectedView === 'visualization' && 'Overlay visualization with lesion boundaries'}
                                    {selectedView === 'confidence' && 'Confidence heatmap showing prediction certainty'}
                                </p>
                            </div>
                        </motion.div>

                        {/* Quick Stats */}
                        <div className="quick-stats">
                            <motion.div
                                className="stat-card glass"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.2 }}
                            >
                                <div className="stat-icon" style={{ background: 'rgba(102, 126, 234, 0.2)' }}>
                                    📊
                                </div>
                                <div className="stat-content">
                                    <span className="stat-label">Lesion Area</span>
                                    <span className="stat-value">{results.metrics.lesion_percentage.toFixed(2)}%</span>
                                </div>
                            </motion.div>

                            <motion.div
                                className="stat-card glass"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.3 }}
                            >
                                <div className="stat-icon" style={{ background: 'rgba(245, 87, 108, 0.2)' }}>
                                    🎯
                                </div>
                                <div className="stat-content">
                                    <span className="stat-label">Confidence</span>
                                    <span className="stat-value">{(results.metrics.mean_confidence * 100).toFixed(1)}%</span>
                                </div>
                            </motion.div>

                            <motion.div
                                className="stat-card glass"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.4 }}
                            >
                                <div className="stat-icon" style={{ background: 'rgba(74, 222, 128, 0.2)' }}>
                                    ✓
                                </div>
                                <div className="stat-content">
                                    <span className="stat-label">Predicted IoU</span>
                                    <span className="stat-value">{(results.metrics.predicted_iou * 100).toFixed(1)}%</span>
                                </div>
                            </motion.div>

                            <motion.div
                                className="stat-card glass"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.5 }}
                            >
                                <div className="stat-icon" style={{ background: 'rgba(167, 139, 250, 0.2)' }}>
                                    📍
                                </div>
                                <div className="stat-content">
                                    <span className="stat-label">Location</span>
                                    <span className="stat-value">{results.metrics.tumor_location || 'N/A'}</span>
                                </div>
                            </motion.div>
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Zoom Modal */}
            {zoomedImage && (
                <motion.div
                    className="zoom-modal"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => setZoomedImage(null)}
                >
                    <div className="zoom-content" onClick={(e) => e.stopPropagation()}>
                        <button className="close-zoom" onClick={() => setZoomedImage(null)}>
                            <X size={24} />
                        </button>
                        <img
                            src={`data:image/png;base64,${zoomedImage.image}`}
                            alt={zoomedImage.label}
                            className="zoomed-image"
                        />
                    </div>
                </motion.div>
            )}
        </section>
    )
}

export default ResultsSection
