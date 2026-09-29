import { Check, CircleAlert } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Sidebar, SidebarContent, SidebarFooter, SidebarMenu, SidebarMenuButton } from '@/components/ui/sidebar'

export type ActiveTab = 'input' | 'stage_1' | 'stage_2' | 'stage_3' | 'stage_4' | 'stage_5' | 'output'
export type PipelineState = 'waiting' | 'running' | 'complete' | 'failed'

type StageLink = { id: string; index: string; shortName: string }

interface AppSidebarProps {
  activeTab: ActiveTab
  onTabChange: (tab: ActiveTab) => void
  stages: StageLink[]
  getStageState: (stageId: string) => PipelineState
  outputState: PipelineState
  productsReady: boolean
  progress: number
  currentStage: string
}

function statusVariant(state: PipelineState) {
  if (state === 'complete') return 'success'
  if (state === 'running') return 'warning'
  if (state === 'failed') return 'destructive'
  return 'muted'
}

function StageMark({ state, index }: { state: PipelineState; index: string }) {
  if (state === 'complete') return <Check aria-hidden="true" />
  if (state === 'failed') return <CircleAlert aria-hidden="true" />
  return <span>{index}</span>
}

export function AppSidebar({
  activeTab,
  onTabChange,
  stages,
  getStageState,
  outputState,
  productsReady,
  progress,
  currentStage,
}: AppSidebarProps) {
  return (
    <Sidebar aria-label="Registration workflow" className="sticky top-16 z-40 max-h-none w-full border-b border-sidebar-border shadow-sm lg:fixed lg:inset-y-16 lg:left-0 lg:h-[calc(100vh-4rem)] lg:max-h-none lg:w-64 lg:border-b-0 lg:border-r lg:shadow-none">
      <SidebarContent className="sidebar-content min-h-0 flex-1 overflow-hidden lg:overflow-y-auto">
        <SidebarMenu aria-label="Pipeline stages" className="flex w-max min-w-full flex-row items-center gap-1.5 overflow-x-auto px-3 py-2 lg:w-full lg:flex-col lg:items-stretch lg:overflow-x-hidden lg:px-3 lg:py-3">
          <div className="sidebar-group-header px-2 pt-1 pb-1 text-[10px] font-mono font-bold tracking-widest text-muted-foreground/70 uppercase hidden lg:block">
            Setup
          </div>
          <SidebarMenuButton className="w-48 shrink-0 lg:w-full" isActive={activeTab === 'input'} onClick={() => onTabChange('input')}>
            <span className="sidebar-num">00</span>
            <span className="sidebar-labels"><strong>Input products</strong></span>
            <Badge variant={productsReady ? 'success' : 'muted'} className="sidebar-badge">{productsReady ? 'Ready' : 'Setup'}</Badge>
          </SidebarMenuButton>

          <Separator className="sidebar-divider mx-1 my-0 h-auto w-px self-stretch lg:mx-2 lg:my-1.5 lg:h-px lg:w-auto" />

          <div className="sidebar-group-header px-2 pt-1 pb-1 text-[10px] font-mono font-bold tracking-widest text-muted-foreground/70 uppercase hidden lg:block">
            Execution Pipeline
          </div>
          {stages.map((stage) => {
            const state = getStageState(stage.id)
            return (
              <SidebarMenuButton
                key={stage.id}
                isActive={activeTab === stage.id}
                onClick={() => onTabChange(stage.id as ActiveTab)}
                className={`w-48 shrink-0 state-${state} lg:w-full`}
              >
                <span className={`sidebar-num stage-num ${state}`}><StageMark state={state} index={stage.index} /></span>
                <span className="sidebar-labels"><strong>{stage.shortName}</strong></span>
                <Badge variant={statusVariant(state)} className="sidebar-badge">
                  {state === 'running' ? 'Running' : state === 'complete' ? 'Done' : state === 'failed' ? 'Failed' : 'Waiting'}
                </Badge>
              </SidebarMenuButton>
            )
          })}

          <Separator className="sidebar-divider mx-1 my-0 h-auto w-px self-stretch lg:mx-2 lg:my-1.5 lg:h-px lg:w-auto" />

          <div className="sidebar-group-header px-2 pt-1 pb-1 text-[10px] font-mono font-bold tracking-widest text-muted-foreground/70 uppercase hidden lg:block">
            Verification & Data
          </div>
          <SidebarMenuButton isActive={activeTab === 'output'} onClick={() => onTabChange('output')} className={`w-48 shrink-0 state-${outputState} lg:w-full`}>
            <span className={`sidebar-num stage-num ${outputState}`}><StageMark state={outputState} index="06" /></span>
            <span className="sidebar-labels"><strong>Output &amp; results</strong></span>
            <Badge variant={statusVariant(outputState)} className="sidebar-badge">
              {outputState === 'complete' ? 'Verified' : outputState === 'running' ? 'Active' : outputState === 'failed' ? 'Failed' : 'Waiting'}
            </Badge>
          </SidebarMenuButton>
        </SidebarMenu>
      </SidebarContent>
      <SidebarFooter>
        <Card className="sidebar-progress-card">
          <div className="sidebar-progress-label">
            <span>Registration Status</span>
            <strong>{progress}%</strong>
          </div>
          <Progress value={progress} aria-label="Registration progress" />
          <p className="sidebar-stage-label">{currentStage}</p>
        </Card>
      </SidebarFooter>
    </Sidebar>
  )
}
