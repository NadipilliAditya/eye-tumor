import { useState } from 'react'
import { motion } from 'framer-motion'
import Header from './components/Header'
import Hero from './components/Hero'
import UploadSection from './components/UploadSection'
import ResultsSection from './components/ResultsSection'
import MetricsDisplay from './components/MetricsDisplay'

import './App.css'

function App() {
    const [results, setResults] = useState(null)
    const [loading, setLoading] = useState(false)

    const handlePredictionComplete = (predictionResults) => {
        setResults(predictionResults)
    }

    const handleReset = () => {
        setResults(null)
    }

    return (
        <div className="app">
            <Header />

            <main className="main-content">
                <Hero />

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                    className="container"
                >
                    <UploadSection
                        onPredictionComplete={handlePredictionComplete}
                        loading={loading}
                        setLoading={setLoading}
                    />

                    {results && (
                        <>
                            <ResultsSection
                                results={results}
                                onReset={handleReset}
                            />

                            <MetricsDisplay metrics={results.metrics} />
                        </>
                    )}
                </motion.div>
            </main>


        </div>
    )
}

export default App
