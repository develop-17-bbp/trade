import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

const PARTICLE_COUNT = 1500
const BOUNDS = { x: 30, y: 20, z: 20 }

function Particles() {
  const ref = useRef<THREE.Points>(null!)

  const { positions, colors, speeds } = useMemo(() => {
    const positions = new Float32Array(PARTICLE_COUNT * 3)
    const colors = new Float32Array(PARTICLE_COUNT * 3)
    const speeds = new Float32Array(PARTICLE_COUNT)

    const palette = [
      new THREE.Color('#00ffcc'),
      new THREE.Color('#00aaff'),
      new THREE.Color('#aa55ff'),
    ]

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3

      positions[i3] = (Math.random() - 0.5) * BOUNDS.x
      positions[i3 + 1] = (Math.random() - 0.5) * BOUNDS.y
      positions[i3 + 2] = (Math.random() - 0.5) * BOUNDS.z

      const color = palette[Math.floor(Math.random() * palette.length)]
      colors[i3] = color.r
      colors[i3 + 1] = color.g
      colors[i3 + 2] = color.b

      speeds[i] = 0.2 + Math.random() * 0.5
    }

    return { positions, colors, speeds }
  }, [])

  const positionsAttr = useMemo(() => {
    const attr = new THREE.BufferAttribute(positions, 3)
    attr.setUsage(THREE.DynamicDrawUsage)
    return attr
  }, [positions])

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const pos = ref.current.geometry.attributes.position as THREE.BufferAttribute
    const array = pos.array as Float32Array

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3
      const speed = speeds[i]

      // Drift upward
      array[i3 + 1] += speed * 0.008

      // Gentle horizontal oscillation
      array[i3] += Math.sin(t * 0.3 + i * 0.1) * 0.003
      array[i3 + 2] += Math.cos(t * 0.2 + i * 0.15) * 0.002

      // Wrap around when particle goes above bounds
      if (array[i3 + 1] > BOUNDS.y * 0.5) {
        array[i3 + 1] = -BOUNDS.y * 0.5
        array[i3] = (Math.random() - 0.5) * BOUNDS.x
        array[i3 + 2] = (Math.random() - 0.5) * BOUNDS.z
      }
    }

    pos.needsUpdate = true
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" {...positionsAttr} />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.4}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
      />
    </points>
  )
}

export default function ParticleField() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none">
      <Canvas
        camera={{ position: [0, 0, 15], fov: 60 }}
        gl={{ alpha: true, antialias: true }}
        dpr={[1, 1.5]}
      >
        <Particles />
      </Canvas>
    </div>
  )
}
