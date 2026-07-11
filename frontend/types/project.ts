import type { TextOverlayStyle } from './project-model'

export interface TextPreset {
  id: string
  name: string
  category: 'titles' | 'lower-thirds' | 'captions' | 'end-cards'
  style: Partial<TextOverlayStyle>
}

export const TEXT_PRESETS: TextPreset[] = [
  { id: 'centered-title', name: 'Centered Title', category: 'titles', style: { text: 'Title', fontSize: 72, fontWeight: 'bold', positionX: 50, positionY: 50, textAlign: 'center' } },
  { id: 'big-bold', name: 'Big & Bold', category: 'titles', style: { text: 'HEADLINE', fontSize: 96, fontWeight: '900', positionX: 50, positionY: 45, textAlign: 'center', letterSpacing: 4 } },
  { id: 'subtitle-style', name: 'Subtitle', category: 'captions', style: { text: 'Subtitle text', fontSize: 36, fontWeight: 'normal', positionX: 50, positionY: 88, textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.6)', padding: 8, borderRadius: 4 } },
  { id: 'lower-third-basic', name: 'Lower Third', category: 'lower-thirds', style: { text: 'Name Here', fontSize: 32, fontWeight: '600', positionX: 10, positionY: 82, textAlign: 'left', backgroundColor: 'rgba(0,0,0,0.7)', padding: 12, borderRadius: 6, maxWidth: 40 } },
  { id: 'lower-third-accent', name: 'Accent Lower Third', category: 'lower-thirds', style: { text: 'Speaker Name', fontSize: 28, fontWeight: '500', positionX: 8, positionY: 85, textAlign: 'left', color: '#FFFFFF', backgroundColor: 'rgba(124,58,237,0.85)', padding: 10, borderRadius: 4, maxWidth: 35 } },
  { id: 'end-card', name: 'End Card', category: 'end-cards', style: { text: 'Thank You', fontSize: 80, fontWeight: '300', positionX: 50, positionY: 45, textAlign: 'center', letterSpacing: 8, color: '#E4E4E7' } },
  { id: 'corner-tag', name: 'Corner Tag', category: 'captions', style: { text: 'LIVE', fontSize: 20, fontWeight: '700', positionX: 92, positionY: 8, textAlign: 'right', color: '#FFFFFF', backgroundColor: 'rgba(239,68,68,0.9)', padding: 6, borderRadius: 4 } },
]
