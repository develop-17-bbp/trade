import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const PARTICLE_COUNT = 800

function Particles() {
  const meshRef = useRef<THREE.Points>(null)

  const [positions, velocities] = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3)
    const vel = new Float32Array(PARTICLE_COUNT * 3)
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3
      pos[i3] = (Math.random() - 0.5) * 40
      pos[i3 + 1] = (Math.random() - 0.5) * 40
      pos[i3 + 2] = (Math.random() - 0.5) * 20
      vel[i3] = (Math.random() - 0.5) * 0.005
      vel[i3 + 1] = (Math.random() - 0.5) * 0.005
      vel[i3 + 2] = (Math.random() - 0.5) * 0.002
    }
    return [pos, vel]
  }, [])

  const colors = useMemo(() => {
    const col = new Float32Array(PARTICLE_COUNT * 3)
    const palette = [
      [0, 1, 0.53],    // green  #00ff88
      [0, 0.67, 1],    // blue   #00aaff
      [0.67, 0.33, 1], // purple #aa55ff
      [0, 1, 0.8],     // cyan   #00ffcc
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

  useFrame(() => {
    if (!meshRef.current) return
    const posAttr = meshRef.current.geometry.attributes.position as THREE.BufferAttribute
    const posArray = posAttr.array as Float32Array

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3
      posArray[i3] += velocities[i3]
      posArray[i3 + 1] += velocities[i3 + 1]
      posArray[i3 + 2] += velocities[i3 + 2]

      // Wrap around bounds
      if (posArray[i3] > 20) posArray[i3] = -20
      if (posArray[i3] < -20) posArray[i3] = 20
      if (posArray[i3 + 1] > 20) posArray[i3 + 1] = -20
      if (posArray[i3 + 1] < -20) posArray[i3 + 1] = 20
      if (posArray[i3 + 2] > 10) posArray[i3 + 2] = -10
      if (posArray[i3 + 2] < -10) posArray[i3 + 2] = 10
    }
    posAttr.needsUpdate = true
  })

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={PARTICLE_COUNT}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={PARTICLE_COUNT}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.4}
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
