import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const PARTICLE_COUNT = 1200

function Particles() {
  const meshRef = useRef<THREE.Points>(null)

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3)
    const vel = new Float32Array(PARTICLE_COUNT * 3)
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3
      pos[i3] = (Math.random() - 0.5) * 50
      pos[i3 + 1] = (Math.random() - 0.5) * 50
      pos[i3 + 2] = (Math.random() - 0.5) * 25
      // Slow upward drift with subtle horizontal sway
      vel[i3] = (Math.random() - 0.5) * 0.003
      vel[i3 + 1] = Math.random() * 0.008 + 0.002  // upward
      vel[i3 + 2] = (Math.random() - 0.5) * 0.001
    }
    return [pos, vel]
  }, [])

  const colors = useMemo(() => {
    const col = new Float32Array(PARTICLE_COUNT * 3)
    // Cyberpunk neon palette: cyan, magenta, violet, hot pink
    const palette = [
      [0, 1, 0.94],     // cyan    #00fff0
      [1, 0, 0.67],     // magenta #ff00aa
      [0.75, 0.27, 1],  // violet  #bf45ff
      [0, 0.8, 1],      // blue    #00ccff
      [0.53, 0.27, 1],  // purple  #8844ff
      [1, 0.13, 0.4],   // hot pink #ff2266
    ]
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3
      const c = palette[Math.floor(Math.random() * palette.length)]
      col[i3] = c[0]
      col[i3 + 1] = c[1]
      col[i3 + 2] = c[2]
    }
    return col
  }, [])

  useFrame((state) => {
    if (!meshRef.current) return
    const posAttr = meshRef.current.geometry.attributes.position as THREE.BufferAttribute
    const posArray = posAttr.array as Float32Array
    const t = state.clock.getElapsedTime()

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3
      // Add gentle sine wave sway
      posArray[i3] += velocities[i3] + Math.sin(t * 0.3 + i * 0.01) * 0.002
      posArray[i3 + 1] += velocities[i3 + 1]
      posArray[i3 + 2] += velocities[i3 + 2]

      // Wrap
      if (posArray[i3] > 25) posArray[i3] = -25
      if (posArray[i3] < -25) posArray[i3] = 25
      if (posArray[i3 + 1] > 25) posArray[i3 + 1] = -25
      if (posArray[i3 + 2] > 12) posArray[i3 + 2] = -12
      if (posArray[i3 + 2] < -12) posArray[i3 + 2] = 12
    }
    posAttr.needsUpdate = true
  })

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={PARTICLE_COUNT} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={PARTICLE_COUNT} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.07}
        vertexColors
        transparent
        opacity={0.5}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

export default function ParticleField() {
  return (
    <Canvas
      camera={{ position: [0, 0, 15], fov: 60 }}
      style={{ width: '100%', height: '100%' }}
      gl={{ alpha: true, antialias: false }}
    >
      <Particles />
    </Canvas>
  )
}
