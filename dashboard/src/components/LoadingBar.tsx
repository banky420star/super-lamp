import React from 'react'

const colors = {
  bg: '#0d1726',
  panelBg: 'rgba(13,23,38,0.92)',
  border: 'rgba(255,255,255,0.08)',
  text: '#eef5ff',
  muted: '#97a9c6',
  cyan: '#5ad7ff',
}

interface LoadingBarProps {
  label?: string
  height?: number
}

export const LoadingBar: React.FC<LoadingBarProps> = ({ label = 'Loading...', height = 4 }) => {
  return (
    <div style={{ padding: '16px 0' }}>
      {label && (
        <div style={{ fontSize: 12, color: colors.muted, marginBottom: 10, textAlign: 'center' }}>
          {label}
        </div>
      )}
      <div
        style={{
          width: '100%',
          height,
          background: 'rgba(90,215,255,0.08)',
          borderRadius: height / 2,
          overflow: 'hidden',
          border: `1px solid ${colors.border}`,
        }}
      >
        <div
          style={{
            width: '40%',
            height: '100%',
            background: `linear-gradient(90deg, ${colors.cyan}00, ${colors.cyan}, ${colors.cyan}00)`,
            borderRadius: height / 2,
            animation: 'loadingBarSlide 1.2s ease-in-out infinite',
          }}
        />
      </div>
      <style>{`
        @keyframes loadingBarSlide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(250%); }
        }
      `}</style>
    </div>
  )
}

export default LoadingBar
