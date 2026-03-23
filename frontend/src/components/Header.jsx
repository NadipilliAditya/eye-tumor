import { motion } from 'framer-motion'
import { Eye, Activity } from 'lucide-react'
import './Header.css'

function Header() {
    return (
        <motion.header
            className="header glass"
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.6, type: 'spring', stiffness: 100 }}
        >
            <div className="container">
                <div className="header-content flex items-center justify-between">
                    <div className="logo flex items-center gap-md">
                        <div className="logo-icon">
                            <Eye size={32} className="icon-primary" />
                        </div>
                        <div>
                            <h1 className="logo-title gradient-text">Ocular Lesion Segmentation</h1>
                            <p className="logo-subtitle">Powered by Few-Shot MedSAM & Prompt Learning</p>
                        </div>
                    </div>

                    <nav className="nav flex items-center gap-md">
                        <motion.div
                            className="status-indicator flex items-center gap-sm"
                            whileHover={{ scale: 1.05 }}
                        >
                            <Activity size={20} className="pulse" style={{ color: '#4ade80' }} />
                            <span className="status-text">System Active</span>
                        </motion.div>
                    </nav>
                </div>
            </div>
        </motion.header>
    )
}

export default Header
