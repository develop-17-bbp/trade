import { Canvas, useFrame, extend } from '@react-three/fiber'
import { OrbitControls, shaderMaterial } from '@react-three/drei'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

interface AIBrainOrbProps {
  consensus: number
  sentiment: 'bullish' | 'bearish' | 'neutral'
  size?: number
}

const SENTIMENT_COLORS = {
  bullish: new THREE.Color('#00ff88'),
  bearish: new THREE.Color('#ff3366'),
  neutral: new THREE.Color('#00aaff'),
} as const

// Custom shader for pulsing glow sphere
const OrbShaderMaterial = shaderMaterial(
  {
    uTime: 0,
    uColor: new THREE.Color('#00ff88'),
    uConsensus: 0.5,
  },
  // Vertex shader
  /* glsl */ `
    varying vec3 vNormal;
    varying vec3 vPosition;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  // Fragment shader
  /* glsl */ `
    uniform float uTime;
    uniform vec3 uColor;
    uniform float uConsensus;
    varying vec3 vNormal;
    varying vec3 vPosition;

    void main() {
      // Fresnel-like rim glow
      float rimPower = 2.0;
      float rim = 1.0 - max(dot(vNormal, vec3(0.0, 0.0, 1.0)), 0.0);
      rim = pow(rim, rimPower);

      // Pulsing intensity
      float pulse = 0.7 + 0.3 * sin(uTime * 2.0) * uConsensus;

      // Surface pattern
      float pattern = sin(vPosition.x * 8.0 + uTime) *
                      sin(vPosition.y * 8.0 + uTime * 0.7) *
                      sin(vPosition.z * 8.0 + uTime * 0.5);
      pattern = smoothstep(-0.2, 0.8, pattern) * 0.3;

      // Combine
      vec3 baseColor = uColor * pulse;
      vec3 rimColor = uColor * rim * 1.5 * uConsensus;
      vec3 patternColor = uColor * pattern * 0.5;

      vec3 finalColor = baseColor * 0.3 + rimColor + patternColor;
      float alpha = 0.6 + rim * 0.4;

      gl_FragColor = vec4(finalColor, alpha);
    }
  `
)

extend({ OrbShaderMaterial })

// Type augmentation for JSX
declare module '@react-three/fiber' {
  interface ThreeElements {
    orbShaderMaterial: {
      ref?: React.Ref<THREE.ShaderMaterial & { uTime: number; uColor: THREE.Color; uConsensus: number }>
      transparent?: boolean
      depthWrite?: boolean
      uTime?: number
      uColor?: THREE.Color
      uConsensus?: number
    }
  }
}

function OrbCore({ consensus, sentiment, size }: Required<AIBrainOrbProps>) {
  const matRef = useRef<THREE.ShaderMaterial & { uTime: number; uColor: THREE.Color; uConsensus: number }>(null!)
  const targetColor = useRef(SENTIMENT_COLORS[sentiment].clone())
  const currentColor = useRef(SENTIMENT_COLORS[sentiment].clone())

  useFrame(({ clock }) => {
    if (!matRef.current) return

    matRef.current.uTime = clock.getElapsedTime()
    matRef.current.uConsensus = consensus

    // Smoothly lerp color
    targetColor.current.copy(SENTIMENT_COLORS[sentiment])
    currentColor.current.lerp(targetColor.current, 0.05)
    matRef.current.uColor = currentColor.current
  })

  return (
    <mesh>
      <sphereGeometry args={[size * 0.45, 64, 64]} />
      <orbShaderMaterial
        ref={matRef}
        transparent
        depthWrite={false}
      />
    </mesh>
  )
}

function Wireframe({ size, consensus }: { size: number; consensus: number }) {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.getElapsedTime()
    ref.current.rotation.y = t * 0.15
    ref.current.rotation.x = Math.sin(t * 0.1) * 0.2
  })

  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[size * 0.55, 1]} />
      <meshBasicMaterial
        wireframe
        color="#ffffff"
        transparent
        opacity={0.08 + consensus * 0.07}
      />
    </mesh>
  )
}

function OrbitingParticles({ size, sentiment, consensus }: Required<AIBrainOrbProps>) {
  const ref = useRef<THREE.Points>(null!)
  const particleCount = 60

  const { positions, basePositions } = useMemo(() => {
    const positions = new Float32Array(particleCount * 3)
    const basePositions = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = size * 0.55 + Math.random() * size * 0.2

      basePositions[i3] = r * Math.sin(phi) * Math.cos(theta)
      basePositions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      basePositions[i3 + 2] = r * Math.cos(phi)

      positions[i3] = basePositions[i3]
      positions[i3 + 1] = basePositions[i3 + 1]
      positions[i3 + 2] = basePositions[i3 + 2]
    }

    return { positions, basePositions }
  }, [size])

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.getElapsedTime()
    const pos = ref.current.geometry.attributes.position as THREE.BufferAttribute
    const arr = pos.array as Float32Array

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3
      const speed = 0.3 + (i / particleCount) * 0.4
      const phase = i * 0.5

      // Orbit around origin
      const cos = Math.cos(t * speed + phase)
      const sin = Math.sin(t * speed + phase)

      const bx = basePositions[i3]
      const bz = basePositions[i3 + 2]

      arr[i3] = bx * cos - bz * sin
      arr[i3 + 1] = basePositions[i3 + 1] + Math.sin(t * 0.5 + phase) * 0.15
      arr[i3 + 2] = bx * sin + bz * cos
    }

    pos.needsUpdate = true
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          usage={THREE.DynamicDrawUsage}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color={SENTIMENT_COLORS[sentiment]}
        transparent
        opacity={0.3 + consensus * 0.5}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
      />
    </points>
  )
}

function GlowHalo({ size, sentiment, consensus }: Required<AIBrainOrbProps>) {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.getElapsedTime()
    const scale = 1 + Math.sin(t * 1.5) * 0.05 * consensus
    ref.current.scale.setScalar(scale)
  })

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[size * 0.6, 32, 32]} />
      <meshBasicMaterial
        color={SENTIMENT_COLORS[sentiment]}
        transparent
        opacity={0.03 + consensus * 0.04}
        side={THREE.BackSide}
      />
    </mesh>
  )
}

export default function AIBrainOrb({
  consensus,
  sentiment,
  size = 3,
}: AIBrainOrbProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, size * 1.8], fov: 50 }}
      gl={{ alpha: true, antialias: true }}
      dpr={[1, 2]}
      style={{ background: 'transparent' }}
    >
      <ambientLight intensity={0.1} />
      <pointLight position={[5, 5, 5]} intensity={0.3} />

      <OrbCore consensus={consensus} sentiment={sentiment} size={size} />
      <Wireframe size={size} consensus={consensus} />
      <OrbitingParticles consensus={consensus} sentiment={sentiment} size={size} />
      <GlowHalo consensus={consensus} sentiment={sentiment} size={size} />

      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </Canvas>
  )
}
