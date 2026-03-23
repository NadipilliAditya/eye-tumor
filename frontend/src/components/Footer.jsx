import { Github, Heart } from 'lucide-react'
import './Footer.css'

function Footer() {
    const currentYear = new Date().getFullYear()

    return (
        <footer className="footer glass">
            <div className="container">
                <div className="footer-content">
                    <div className="footer-section">
                        <h3 className="footer-title gradient-text">MedSAM Vision</h3>
                        <p className="footer-description">
                            Advanced ocular lesion segmentation using few-shot learning and prompt engineering techniques.
                        </p>
                    </div>

                    <div className="footer-section">
                        <h4 className="footer-heading">Technology</h4>
                        <ul className="footer-list">
                            <li>MedSAM (Medical SAM)</li>
                            <li>Few-Shot Learning</li>
                            <li>Prompt Learning</li>
                            <li>Deep Learning</li>
                        </ul>
                    </div>

                    <div className="footer-section">
                        <h4 className="footer-heading">Metrics</h4>
                        <ul className="footer-list">
                            <li>Accuracy & Dice Score</li>
                            <li>IoU & Precision</li>
                            <li>Recall & F1 Score</li>
                            <li>Confidence Maps</li>
                        </ul>
                    </div>

                    <div className="footer-section">
                        <h4 className="footer-heading">Resources</h4>
                        <ul className="footer-list">
                            <li><a href="#" className="footer-link">Documentation</a></li>
                            <li><a href="#" className="footer-link">API Reference</a></li>
                            <li><a href="#" className="footer-link">Research Paper</a></li>
                            <li><a href="#" className="footer-link">GitHub</a></li>
                        </ul>
                    </div>
                </div>

                <div className="footer-bottom">
                    <div className="footer-copyright">
                        <span>© {currentYear} MedSAM Vision. Made with </span>
                        <Heart size={16} className="heart-icon" fill="currentColor" />
                        <span> for medical AI</span>
                    </div>
                    <div className="footer-links">
                        <a href="#" className="footer-icon-link" title="GitHub">
                            <Github size={20} />
                        </a>
                    </div>
                </div>
            </div>
        </footer>
    )
}

export default Footer
