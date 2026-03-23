import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Image as ImageIcon, Loader, AlertCircle } from 'lucide-react'
import axios from 'axios'
import './UploadSection.css'

function UploadSection({ onPredictionComplete, loading, setLoading }) {
    const [selectedFile, setSelectedFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [threshold, setThreshold] = useState(0.5)
    const [error, setError] = useState(null)
    const [dragActive, setDragActive] = useState(false)
    const fileInputRef = useRef(null)

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file)
            setError(null)

            // Create preview
            const reader = new FileReader()
            reader.onloadend = () => {
                setPreview(reader.result)
            }
            reader.readAsDataURL(file)
        } else {
            setError('Please select a valid image file')
        }
    }

    const handleDrag = (e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true)
        } else if (e.type === 'dragleave') {
            setDragActive(false)
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0])
        }
    }

    const handleFileInput = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0])
        }
    }

    const handleSubmit = async () => {
        if (!selectedFile) {
            setError('Please select an image first')
            return
        }

        setLoading(true)
        setError(null)

        const formData = new FormData()
        formData.append('image', selectedFile)
        formData.append('threshold', threshold)

        try {
            const response = await axios.post('/api/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            })

            if (response.data.success) {
                onPredictionComplete(response.data)
            } else {
                setError(response.data.error || 'Prediction failed')
            }
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to connect to server. Please ensure the backend is running.')
        } finally {
            setLoading(false)
        }
    }

    const handleReset = () => {
        setSelectedFile(null)
        setPreview(null)
        setError(null)
        if (fileInputRef.current) {
            fileInputRef.current.value = ''
        }
    }

    return (
        <section className="upload-section">
            <div className="container">
                <motion.div
                    className="upload-container"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <h2 className="section-title">Upload Eye Image</h2>
                    <p className="section-description">
                        Upload a medical eye image to detect and segment ocular lesions
                    </p>

                    <div className="upload-grid">
                        {/* Upload Area */}
                        <div className="upload-area-container">
                            <div
                                className={`upload-area glass ${dragActive ? 'drag-active' : ''} ${preview ? 'has-preview' : ''}`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                                onClick={() => !preview && fileInputRef.current?.click()}
                            >
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileInput}
                                    style={{ display: 'none' }}
                                />

                                <AnimatePresence mode="wait">
                                    {preview ? (
                                        <motion.div
                                            key="preview"
                                            className="preview-container"
                                            initial={{ opacity: 0, scale: 0.9 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.9 }}
                                        >
                                            <img src={preview} alt="Preview" className="preview-image" />
                                            <motion.button
                                                className="btn btn-secondary change-btn"
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    handleReset()
                                                }}
                                                whileHover={{ scale: 1.05 }}
                                                whileTap={{ scale: 0.95 }}
                                            >
                                                Change Image
                                            </motion.button>
                                        </motion.div>
                                    ) : (
                                        <motion.div
                                            key="upload"
                                            className="upload-placeholder"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                        >
                                            <div className="upload-icon">
                                                <Upload size={48} />
                                            </div>
                                            <h3>Drop your image here</h3>
                                            <p>or click to browse</p>
                                            <div className="supported-formats">
                                                <ImageIcon size={16} />
                                                <span>JPG, PNG, JPEG supported</span>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            {error && (
                                <motion.div
                                    className="error-message"
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                >
                                    <AlertCircle size={20} />
                                    <span>{error}</span>
                                </motion.div>
                            )}
                        </div>

                        {/* Controls */}
                        <div className="controls-container">
                            <div className="control-card glass">
                                <h3 className="control-title">Segmentation Settings</h3>

                                <div className="control-group">
                                    <label className="control-label">
                                        <span>Threshold</span>
                                        <span className="threshold-value">{threshold.toFixed(2)}</span>
                                    </label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={threshold}
                                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                        className="threshold-slider"
                                    />
                                    <div className="threshold-labels">
                                        <span>Less Sensitive</span>
                                        <span>More Sensitive</span>
                                    </div>
                                </div>

                                <motion.button
                                    className="btn btn-primary analyze-btn"
                                    onClick={handleSubmit}
                                    disabled={!selectedFile || loading}
                                    whileHover={!loading && selectedFile ? { scale: 1.02 } : {}}
                                    whileTap={!loading && selectedFile ? { scale: 0.98 } : {}}
                                >
                                    {loading ? (
                                        <>
                                            <Loader className="spinner-icon" size={20} />
                                            <span>Analyzing...</span>
                                        </>
                                    ) : (
                                        <>
                                            <ImageIcon size={20} />
                                            <span>Analyze Image</span>
                                        </>
                                    )}
                                </motion.button>
                            </div>

                            <div className="info-card glass">
                                <h4>How it works</h4>
                                <ol className="info-list">
                                    <li>Upload an eye medical image</li>
                                    <li>Adjust the detection threshold</li>
                                    <li>Click "Analyze Image"</li>
                                    <li>View segmentation results and metrics</li>
                                </ol>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    )
}

export default UploadSection
