import { motion } from 'framer-motion'
import { Sparkles, Target, Zap } from 'lucide-react'
import './Hero.css'

function Hero() {
    const features = [
        { icon: Target, text: 'Few-Shot Learning', color: '#667eea' },
        { icon: Zap, text: 'Real-time Analysis', color: '#f5576c' },
        { icon: Sparkles, text: 'High Accuracy', color: '#4ade80' },
    ]

    return (
        <section className="hero">
            <div className="container">
                <motion.div
                    className="hero-content text-center"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                >
                    <motion.div
                        className="hero-badge"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                    >
                        <Sparkles size={16} />
                        <span>Powered by MedSAM & Prompt Learning</span>
                    </motion.div>

                    <h1 className="hero-title">
                        Advanced <span className="gradient-text">Ocular Lesion</span>
                        <br />
                        Detection & Segmentation
                    </h1>

                    <p className="hero-description">
                        AI-powered medical image analysis using few-shot learning techniques.
                        Upload eye images to detect and highlight tumors and lesions with
                        comprehensive evaluation metrics.
                    </p>

                    <div className="features-grid">
                        {features.map((feature, index) => (
                            <motion.div
                                key={index}
                                className="feature-card glass"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.4 + index * 0.1 }}
                                whileHover={{ y: -5, boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)' }}
                            >
                                <div
                                    className="feature-icon"
                                    style={{ background: `${feature.color}20`, color: feature.color }}
                                >
                                    <feature.icon size={24} />
                                </div>
                                <span className="feature-text">{feature.text}</span>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            </div>
        </section>
    )
}

export default Hero
